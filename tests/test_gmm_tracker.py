"""Unit tests for GMMUncertaintyTracker.

Tests the math (GMM fitting, log-L thresholds, saturation detection) without
requiring a real MACE calculator. We monkey-patch ``get_embeddings_single``
to return pre-specified vectors so we can drive the tracker deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms


@pytest.fixture
def patch_embeddings(monkeypatch):
    """Replace get_embeddings_single with a function that reads a tag off the
    atoms object: ``atoms.info["_embed"]`` → the returned embedding.
    """
    def fake(atoms, calc, flatten=True, flatten_axis=0):
        vec = atoms.info.get("_embed")
        if vec is None:
            raise ValueError("test: atoms missing '_embed' info key")
        return np.asarray(vec, dtype=np.float64)

    import mcmc.calculators.calculators as cmod
    import mcmc.uncertainty.gmm_tracker as tmod
    monkeypatch.setattr(cmod, "get_embeddings_single", fake)
    # the gmm_tracker imports it locally inside methods, so the patch is
    # picked up through cmod's namespace
    yield


def _toy_atoms(vec):
    """Build a trivial Atoms object carrying an embedding via info dict."""
    a = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    a.info["_embed"] = list(vec)
    return a


def test_gmm_tracker_fit_and_threshold(patch_embeddings):
    """Fit on a cluster of training points; threshold is the 5% quantile."""
    from mcmc.uncertainty.gmm_tracker import GMMUncertaintyTracker

    rng = np.random.default_rng(0)
    # 30 training embeddings, 4-dim, centered at origin
    train = [_toy_atoms(rng.normal(0, 0.3, size=4)) for _ in range(30)]

    tracker = GMMUncertaintyTracker(
        calc=object(),  # dummy (patched get_embeddings_single ignores it)
        training_atoms=train,
        n_components=3,
    )
    assert tracker.gmm is not None
    assert tracker.logl_threshold is not None
    # Auto-threshold must be finite and lower than the training mean log-L
    train_emb = np.stack([a.info["_embed"] for a in train])
    train_logl = tracker.gmm.score_samples(train_emb)
    assert tracker.logl_threshold < float(train_logl.mean())


def test_gmm_tracker_flags_far_points_as_ood(patch_embeddings):
    """A point far from the training cluster should be flagged OOD."""
    from mcmc.uncertainty.gmm_tracker import GMMUncertaintyTracker

    rng = np.random.default_rng(1)
    train = [_toy_atoms(rng.normal(0, 0.3, size=4)) for _ in range(30)]
    tracker = GMMUncertaintyTracker(calc=object(), training_atoms=train, n_components=3)

    # In-distribution snapshot
    logl_in, is_ood_in = tracker.add_snapshot(_toy_atoms(rng.normal(0, 0.3, size=4)))
    # Out-of-distribution snapshot: far away in embedding space
    logl_out, is_ood_out = tracker.add_snapshot(_toy_atoms([10.0, 10.0, 10.0, 10.0]))

    assert logl_in > logl_out
    assert is_ood_out is True
    assert tracker.ood_queue  # should contain the far point


def test_gmm_tracker_saturation_detects_stability(patch_embeddings):
    """If the stream of log-L values stops hitting new lows, saturated."""
    from mcmc.uncertainty.gmm_tracker import GMMUncertaintyTracker

    rng = np.random.default_rng(2)
    train = [_toy_atoms(rng.normal(0, 0.3, size=4)) for _ in range(30)]
    tracker = GMMUncertaintyTracker(
        calc=object(),
        training_atoms=train,
        n_components=3,
        saturation_window=3,
        saturation_tolerance=0.1,
        min_saves_before_check=6,
    )

    # 6 snapshots all near training cluster → log-L stays high, no new lows
    for _ in range(6):
        tracker.add_snapshot(_toy_atoms(rng.normal(0, 0.3, size=4)))

    # Should report saturated (no new lows in the last 3 vs previous 3)
    # Because all 6 log-L's are in the same distribution.
    # Not strict — occasional noise can miss. Test with deterministic stream:
    tracker.history = [-100, -120, -110, -115, -118, -117]  # no new low past -120
    assert tracker.is_saturated() is True


def test_gmm_tracker_saturation_false_when_still_finding_lows(patch_embeddings):
    """If recent snapshots DID turn up new lower log-L, NOT saturated."""
    from mcmc.uncertainty.gmm_tracker import GMMUncertaintyTracker

    rng = np.random.default_rng(3)
    train = [_toy_atoms(rng.normal(0, 0.3, size=4)) for _ in range(30)]
    tracker = GMMUncertaintyTracker(
        calc=object(),
        training_atoms=train,
        n_components=3,
        saturation_window=3,
        saturation_tolerance=0.1,
        min_saves_before_check=6,
    )

    # Latest window drops to new low → not saturated
    tracker.history = [-100, -110, -105, -120, -125, -130]
    assert tracker.is_saturated() is False


def test_gmm_tracker_summary_structure(patch_embeddings):
    from mcmc.uncertainty.gmm_tracker import GMMUncertaintyTracker

    rng = np.random.default_rng(4)
    train = [_toy_atoms(rng.normal(0, 0.3, size=4)) for _ in range(30)]
    tracker = GMMUncertaintyTracker(calc=object(), training_atoms=train, n_components=3)

    tracker.add_snapshot(_toy_atoms(rng.normal(0, 0.3, size=4)))
    tracker.add_snapshot(_toy_atoms([10, 10, 10, 10]))

    s = tracker.summary()
    assert set(s.keys()) >= {
        "n_snapshots", "n_ood", "logl_min", "logl_max", "logl_mean",
        "logl_threshold", "ood_queue_size",
    }
    assert s["n_snapshots"] == 2
    assert s["n_ood"] >= 1
