"""GMM-based out-of-distribution tracker for active-learning early stop.

Single-MACE-model uncertainty (no ensemble needed). Fits a Gaussian Mixture
Model on the latent-space embeddings of the training set at startup. For each
newly-saved MC snapshot, computes the GMM log-likelihood — low log-L means
the structure is far from any training-set cluster in embedding space, i.e.
MACE has not been trained on that kind of configuration and its predictions
are likely unreliable.

Active-learning criterion: the MC run can be stopped early once the log-L of
recent snapshots has "saturated" — i.e. the window of recent saves is no
longer turning up new lower log-L values. Physically this means MC has
stopped discovering configurations outside the training distribution; more
sampling gives diminishing AL return.

Integration points:
  * HamiltonianREMC.__init__ now accepts ``gmm_tracker=GMMUncertaintyTracker(...)``
  * HamiltonianREMC.run calls ``gmm_tracker.add_snapshot(atoms)`` after each save
  * Run breaks when ``gmm_tracker.is_saturated()`` returns True

References:
  * Latent-space OOD detection for MLPs: Sivaraman & Colón 2023 JCTC
  * GMM on embeddings: standard anomaly-detection pattern (sklearn docs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from ase.atoms import Atoms


@dataclass
class GMMUncertaintyTracker:
    """Track MC snapshots' embedding log-likelihood against a training-set GMM.

    Args:
        calc: MACE calculator exposing latent embeddings via
            ``get_embeddings_single``. Typically the same calculator used
            by the MC replicas.
        training_atoms: List of ASE ``Atoms`` representing the MACE training
            set (usually ~tens to ~hundreds of DFT-relaxed structures).
            GMM is fit once on their mean-pooled MACE embeddings.
        n_components: Number of GMM components. Rule of thumb: √N_training.
        logl_threshold: Log-L value below which a snapshot is flagged as OOD.
            If None, auto-set to the 5th percentile of training log-L
            (matches the "5% tail" anomaly convention).
        saturation_window: Number of recent save events to compare against
            when testing saturation. A larger window makes the test slower
            to trigger but more robust.
        saturation_tolerance: A save event is considered to not "find a new
            low" if its log-L is within this tolerance of the existing minimum.
        min_saves_before_check: Don't trigger saturation stop before this
            many save events have occurred (burn-in for the tracker itself).
    """

    calc: object
    training_atoms: list[Atoms]
    n_components: int = 16
    logl_threshold: float | None = None
    saturation_window: int = 3
    saturation_tolerance: float = 0.1
    min_saves_before_check: int = 6
    logger: logging.Logger | None = None

    # runtime state
    gmm: object = field(default=None, init=False, repr=False)
    history: list[float] = field(default_factory=list, init=False, repr=False)
    ood_queue: list[Atoms] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
        self._fit()

    # ----- fitting / setup -------------------------------------------------
    def _fit(self) -> None:
        """Fit GMM on training embeddings. Auto-compute threshold if not set."""
        from sklearn.mixture import GaussianMixture
        from mcmc.calculators.calculators import get_embeddings_single

        if not self.training_atoms:
            raise ValueError("GMMUncertaintyTracker needs a non-empty training_atoms list")

        embeddings = np.stack(
            [get_embeddings_single(a, self.calc, flatten=True) for a in self.training_atoms]
        )
        n = len(self.training_atoms)
        # Cap components at N/2 to avoid degenerate fits with few training points
        k = max(1, min(self.n_components, n // 2))
        self.gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-4,
            max_iter=200,
            random_state=0,
        ).fit(embeddings)

        train_logl = self.gmm.score_samples(embeddings)
        if self.logl_threshold is None:
            self.logl_threshold = float(np.percentile(train_logl, 5.0))

        self.logger.info(
            "GMM tracker: fit on %d training structures → %d components. "
            "Training log-L quantiles: min=%.2f, 5%%=%.2f, 50%%=%.2f, 95%%=%.2f, max=%.2f. "
            "OOD threshold = %.2f.",
            n, k,
            float(train_logl.min()),
            float(np.percentile(train_logl, 5)),
            float(np.percentile(train_logl, 50)),
            float(np.percentile(train_logl, 95)),
            float(train_logl.max()),
            self.logl_threshold,
        )

    # ----- streaming updates ----------------------------------------------
    def add_snapshot(self, atoms: Atoms) -> tuple[float, bool]:
        """Score one snapshot; return (logL, is_OOD). Appends to history.

        Queues the structure for AL relabeling if log-L is below threshold.
        """
        from mcmc.calculators.calculators import get_embeddings_single

        emb = get_embeddings_single(atoms, self.calc, flatten=True)
        logl = float(self.gmm.score_samples(emb.reshape(1, -1))[0])
        self.history.append(logl)

        is_ood = logl < self.logl_threshold
        if is_ood:
            self.ood_queue.append(atoms.copy())
            self.logger.info(
                "[GMM] snapshot %d log-L=%.2f < threshold %.2f → queued "
                "for AL relabel (queue size %d)",
                len(self.history), logl, self.logl_threshold, len(self.ood_queue),
            )
        else:
            self.logger.debug(
                "[GMM] snapshot %d log-L=%.2f ≥ threshold %.2f (in-distribution)",
                len(self.history), logl, self.logl_threshold,
            )
        return logl, is_ood

    # ----- saturation test --------------------------------------------------
    def is_saturated(self) -> bool:
        """Check if recent snapshots have stopped finding new OOD lows.

        True when:
          (a) at least ``min_saves_before_check`` snapshots have been added,
          (b) for the last ``saturation_window`` snapshots, no new minimum
              log-L more than ``saturation_tolerance`` below the prior
              window's minimum has been found.

        Physically: MC is no longer discovering configurations outside the
        training distribution; AL round is "done" — further sampling only
        adds more of the same kind of structure to the pool.
        """
        n = len(self.history)
        if n < max(self.min_saves_before_check, 2 * self.saturation_window):
            return False

        recent = self.history[-self.saturation_window :]
        older = self.history[-2 * self.saturation_window : -self.saturation_window]

        recent_min = min(recent)
        older_min = min(older)
        if recent_min >= older_min - self.saturation_tolerance:
            self.logger.info(
                "[GMM] saturation: recent-min log-L %.2f ≥ older-min %.2f − tol %.2f "
                "→ MC no longer finds lower-log-L (more OOD) configurations. "
                "OOD queue size = %d.",
                recent_min, older_min, self.saturation_tolerance, len(self.ood_queue),
            )
            return True
        return False

    # ----- reporting --------------------------------------------------------
    def summary(self) -> dict:
        """Return a summary of the AL round's uncertainty trace."""
        if not self.history:
            return {"n_snapshots": 0, "n_ood": 0, "ood_queue": []}
        hist = np.array(self.history)
        return {
            "n_snapshots": len(hist),
            "n_ood": int((hist < self.logl_threshold).sum()),
            "logl_min": float(hist.min()),
            "logl_max": float(hist.max()),
            "logl_mean": float(hist.mean()),
            "logl_threshold": self.logl_threshold,
            "ood_queue_size": len(self.ood_queue),
        }
