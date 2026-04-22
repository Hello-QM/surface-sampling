"""Hamiltonian Replica Exchange MC over (T, pH, phi) space for surface Pourbaix diagrams.

Each replica runs at a different (temperature, pH, potential) condition. Swaps between
replicas use the generalized Hamiltonian exchange criterion:

    P(swap) = min(1, exp(-Delta))
    Delta = beta_i * Omega_i(x_j) + beta_j * Omega_j(x_i)
          - beta_i * Omega_i(x_i) - beta_j * Omega_j(x_j)

where Omega is the Pourbaix grand potential at the replica's (pH, phi) conditions,
and beta = 1/kT_mc is the inverse MC sampling temperature.

Important: The MC sampling temperature (temperature) is decoupled from the
electrochemical temperature (electrochemical_temp) used in the Nernst equation.
The MC temperature controls Metropolis acceptance (fictitious, 0.1-1.0 eV),
while electrochemical_temp is fixed at 298K (0.0257 eV) for correct Pourbaix
thermodynamics.

Key insight: DeltaG1 (MACE surface energy) is independent of (pH, phi), so swaps
between replicas at different electrochemical conditions only require recomputing
DeltaG2 -- a cheap analytical sum over atom compositions. No extra MACE evaluations
needed for swaps.

Optionally integrates MACE ensemble uncertainty tracking for on-the-fly active learning:
structures with high force uncertainty are queued for DFT labeling and model fine-tuning.
"""

import copy
import logging
import pickle
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.optimize import FIRE

from mcmc.events.criterion import MetropolisCriterion
from mcmc.events.event import Change, Exchange
from mcmc.events.proposal import ChangeProposal, FixedActionChangeProposal, SwitchProposal
from mcmc.pourbaix.atoms import PourbaixAtom, generate_pourbaix_atoms
from mcmc.system import SurfaceSystem


# Geometric gates applied before the Metropolis energy check. If the
# proposed configuration fails the min-pair-distance gate we reject
# immediately, avoiding a wasted MACE evaluation on a clearly pathological
# structure.
_MC_MIN_PAIR_DIST = 0.9        # Å — anything closer is a collision
_MC_RELAX_FMAX    = 0.5        # eV/Å — stop a brief relax when forces below this
_MC_RELAX_STEPS   = 10         # cap on FIRE steps per accepted move
_MC_ADS_RELAX_Z   = 0.3        # Å — relax atoms more than this above z_top_base


# ──────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReplicaCondition:
    """Electrochemical + thermal condition for one replica.

    Attributes:
        temperature: MC sampling temperature in eV (fictitious, controls
            Metropolis acceptance, typically 0.1-1.0 eV for exploration).
        pH: pH value for the Pourbaix diagram.
        phi: Electrode potential (V vs SHE).
        electrochemical_temp: Physical temperature for Nernst equation and
            electrochemical thermodynamics. Fixed at 298K (0.0257 eV) for
            standard Pourbaix diagrams. Decoupled from MC temperature to
            allow high-temperature sampling without distorting the
            electrochemical free energies.
    """
    temperature: float    # MC sampling temperature in eV (fictitious, ~0.1-1.0 eV)
    pH: float
    phi: float            # V vs SHE
    electrochemical_temp: float = 0.0257  # Physical temperature for Nernst (298K, fixed)

    def __repr__(self):
        return (f"(T_mc={self.temperature:.4f}, T_echem={self.electrochemical_temp:.4f}, "
                f"pH={self.pH:.1f}, φ={self.phi:.2f})")


@dataclass
class SwapStats:
    """Track swap acceptance between specific replica pairs."""
    attempted: int = 0
    accepted: int = 0

    @property
    def rate(self) -> float:
        return self.accepted / max(self.attempted, 1)


# ──────────────────────────────────────────────────────────────
#  Uncertainty tracker for on-the-fly active learning
# ──────────────────────────────────────────────────────────────

class UncertaintyTracker:
    """Track MACE ensemble uncertainty and queue high-uncertainty structures.

    When forces_std exceeds the threshold, the structure is added to a training
    queue. Once the queue reaches `batch_size`, a fine-tuning callback is triggered.

    Args:
        threshold: Force uncertainty threshold (eV/A) to flag a structure.
        batch_size: Number of flagged structures to accumulate before fine-tuning.
        on_batch_ready: Callback(list[Atoms]) invoked when batch is full.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        batch_size: int = 20,
        on_batch_ready=None,
        logger: logging.Logger | None = None,
    ):
        self.threshold = threshold
        self.batch_size = batch_size
        self.on_batch_ready = on_batch_ready
        self.logger = logger or logging.getLogger(__name__)
        self._queue = []
        self._total_flagged = 0

    def check(self, surface: SurfaceSystem) -> bool:
        """Check uncertainty of current structure. Returns True if flagged."""
        results = surface.results if hasattr(surface, 'results') else {}
        forces_std = results.get("forces_std", 0.0)

        if forces_std > self.threshold:
            atoms = surface.real_atoms.copy()
            self._queue.append(atoms)
            self._total_flagged += 1
            self.logger.info(
                "Uncertainty %.4f > %.4f — queued structure %d (batch %d/%d)",
                forces_std, self.threshold, self._total_flagged,
                len(self._queue), self.batch_size,
            )

            if len(self._queue) >= self.batch_size and self.on_batch_ready:
                batch = self._queue.copy()
                self._queue.clear()
                self.on_batch_ready(batch)
            return True
        return False

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    @property
    def total_flagged(self) -> int:
        return self._total_flagged


# ──────────────────────────────────────────────────────────────
#  Core: Hamiltonian Replica Exchange MC
# ──────────────────────────────────────────────────────────────

def compute_dG2(
    atom_symbols: list[str],
    pourbaix_atoms: dict[str, PourbaixAtom],
    pH: float,
    phi: float,
    temperature: float = 0.0257,
) -> float:
    """Compute total ΔG₂ analytically for a given atom composition and conditions.

    This is the key function that enables cheap Hamiltonian swaps: we can evaluate
    the Pourbaix potential of a configuration at arbitrary (pH, φ) without running MACE.

    Args:
        atom_symbols: Chemical symbols of all atoms in the slab.
        pourbaix_atoms: Dict of PourbaixAtom objects for each element.
        pH: pH value.
        phi: Electrode potential (V vs SHE).
        temperature: kT in eV (default 0.0257 = 298K).

    Returns:
        float: Total ΔG₂ summed over all atoms.
    """
    dG2 = 0.0
    for sym in atom_symbols:
        if sym not in pourbaix_atoms:
            continue
        pa = pourbaix_atoms[sym]
        dG2_individual = (
            pa.delta_G2_std
            - pa.num_e * phi
            - np.log(10) * pa.num_H * temperature * pH
            + temperature * np.log(pa.species_conc)
        )
        dG2 += dG2_individual
    return dG2


def compute_dG1(
    atom_symbols: list[str],
    pourbaix_atoms: dict[str, PourbaixAtom],
    slab_energy: float,
) -> float:
    """Compute ΔG₁ (dissociation energy).

    Args:
        atom_symbols: Chemical symbols of all atoms.
        pourbaix_atoms: Dict of PourbaixAtom objects.
        slab_energy: MACE potential energy of the slab.

    Returns:
        float: ΔG₁ = Σμ_std - E_slab.
    """
    sum_chem_pots = 0.0
    for sym in atom_symbols:
        if sym in pourbaix_atoms:
            sum_chem_pots += pourbaix_atoms[sym].atom_std_state_energy
    return sum_chem_pots - slab_energy


class HamiltonianREMC:
    """Hamiltonian Replica Exchange MC across (T, pH, φ) space.

    Builds a complete surface Pourbaix diagram in a single simulation by running
    replicas at different electrochemical conditions and exchanging configurations.

    Args:
        conditions: List of ReplicaCondition(temperature, pH, phi) for each replica.
        adsorbates: List of adsorbate species (e.g. ["O", "HO", "Ir"]).
        canonical: If True, use canonical (swap) moves.
        phase_diagram_path: Path to pymatgen PhaseDiagram JSON.
        pourbaix_diagram_path: Path to pymatgen PourbaixDiagram JSON.
        elements: List of element symbols for Pourbaix atom generation.
        swap_interval: Attempt swaps every N sweeps.
        uncertainty_tracker: Optional UncertaintyTracker for active learning.
    """

    def __init__(
        self,
        conditions: list[ReplicaCondition],
        adsorbates: list[str],
        phase_diagram_path: str,
        pourbaix_diagram_path: str,
        elements: list[str],
        canonical: bool = False,
        swap_interval: int = 1,
        n_trials: int = 1,
        uncertainty_tracker: UncertaintyTracker | None = None,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        if len(conditions) < 2:
            raise ValueError("Need at least 2 replicas.")

        self.conditions = conditions
        self.n_replicas = len(conditions)
        self.adsorbates = adsorbates
        self.canonical = canonical
        self.swap_interval = swap_interval
        # Multiple-Try Metropolis: number of candidate moves evaluated per
        # MC step. 1 = plain Metropolis (original behavior). Larger values
        # (typically 4-8) evaluate k candidates, pick one by Boltzmann
        # weight, and accept via the MTM ratio (Liu 2000). Trades k× more
        # MACE / FIRE calls per step for better phase-space mixing,
        # especially near phase boundaries. Only applies to semigrand
        # (non-canonical) mode; canonical SwitchProposal stays single-try.
        self.n_trials = max(1, int(n_trials))
        self.uncertainty_tracker = uncertainty_tracker
        self.logger = logger or logging.getLogger(__name__)

        # Generate pourbaix_atoms for each unique (pH, φ) condition
        self.logger.info("Generating Pourbaix atoms for %d conditions...", self.n_replicas)
        self.replica_pourbaix_atoms: list[dict[str, PourbaixAtom]] = []
        self._pbx_cache: dict[tuple[float, float], dict[str, PourbaixAtom]] = {}

        for cond in conditions:
            key = (cond.pH, cond.phi)
            if key not in self._pbx_cache:
                pa = generate_pourbaix_atoms(
                    phase_diagram_path, pourbaix_diagram_path,
                    phi=cond.phi, pH=cond.pH, elements=elements,
                )
                self._pbx_cache[key] = pa
                self.logger.info("  %s: %s", cond, {k: v.dominant_species for k, v in pa.items()})
            self.replica_pourbaix_atoms.append(self._pbx_cache[key])

        # Build swap topology: adjacent pairs that can exchange
        self._swap_pairs = self._build_swap_pairs()
        self.logger.info("Swap pairs: %d", len(self._swap_pairs))

        # Track swap statistics per pair
        self._swap_stats: dict[tuple[int, int], SwapStats] = {
            pair: SwapStats() for pair in self._swap_pairs
        }

        self.replicas: list[SurfaceSystem] = []
        self._cached_dG1: dict[int, float] = {}  # replica_idx -> cached ΔG₁

    def _build_swap_pairs(self) -> list[tuple[int, int]]:
        """Build swap pairs: two replicas can swap if they differ in exactly one
        dimension (T, pH, or φ) and are adjacent in that dimension."""
        pairs = []
        for i in range(self.n_replicas):
            for j in range(i + 1, self.n_replicas):
                ci, cj = self.conditions[i], self.conditions[j]
                diffs = (
                    (ci.temperature != cj.temperature),
                    (ci.pH != cj.pH),
                    (ci.phi != cj.phi),
                )
                # Allow swap if conditions differ in exactly 1 dimension
                if sum(diffs) == 1:
                    pairs.append((i, j))
        # If no single-dimension pairs found (arbitrary grid), allow all adjacent
        if not pairs:
            for i in range(self.n_replicas - 1):
                pairs.append((i, i + 1))
        return pairs

    def _create_replicas(self, surface: SurfaceSystem) -> list[SurfaceSystem]:
        """Create independent surface copies, each with its own Pourbaix calculator.

        The calculator temperature is set to electrochemical_temp (298K = 0.0257 eV),
        NOT the MC sampling temperature. This ensures the Nernst equation in
        get_delta_G2 uses the correct physical temperature for electrochemical
        thermodynamics, regardless of the fictitious MC temperature used for
        Metropolis sampling.
        """
        replicas = []
        for i in range(self.n_replicas):
            replica = surface.copy(copy_calc=False)
            # Each replica gets a deep copy of the calculator with its own conditions
            calc = copy.deepcopy(surface.calc)
            calc.set(
                pourbaix_atoms=self.replica_pourbaix_atoms[i],
                pH=self.conditions[i].pH,
                phi=self.conditions[i].phi,
                temperature=self.conditions[i].electrochemical_temp,  # Nernst: always 298K
            )
            replica.set_calc(calc)
            replicas.append(replica)
        return replicas

    def _mc_step(self, replica_idx: int) -> bool:
        """Dispatch an MC step. Uses Multiple-Try Metropolis (n_trials > 1)
        only for semigrand (non-canonical) mode; canonical + MTM is not
        implemented and falls back to single-try SwitchProposal."""
        if self.n_trials > 1 and not self.canonical:
            return self._mc_step_mtm(replica_idx)
        return self._mc_step_single(replica_idx)

    def _mc_step_single(self, replica_idx: int) -> bool:
        """Perform one single-try MC step on a replica (original behavior).

        Flow:
          1. propose a change (add/remove/swap adsorbate)
          2. forward() mutates the system and saves "before" and "after" states
          3. geometric overlap gate — reject immediately if any pair < 0.9 Å
          4. short MACE-based FIRE relaxation of adsorbate atoms only
             (keeps the slab rigid, caps at _MC_RELAX_STEPS)
          5. Metropolis acceptance on the relaxed energy

        The relax step is what prevents "floating" adsorbates: bridge sites
        are now placed geometrically-correct (Ir-O ~ 2.0 Å), and any residual
        mis-placement is polished by FIRE before the energy is evaluated.
        """
        replica = self.replicas[replica_idx]
        temp = self.conditions[replica_idx].temperature

        if self.canonical:
            proposal = SwitchProposal(system=replica, adsorbate_list=self.adsorbates.copy())
            event = Exchange(replica, proposal, MetropolisCriterion(temp))
        else:
            proposal = ChangeProposal(system=replica, adsorbate_list=self.adsorbates.copy())
            event = Change(replica, proposal, MetropolisCriterion(temp))

        # Step 1-2: propose, mutate, save "before"/"after"
        event.forward()

        # Step 3: geometric overlap gate — skip the expensive energy eval if
        # the proposed structure is pathological on its face.
        if not self._geometry_ok(replica.real_atoms):
            event.backward()
            self.logger.debug("MC move rejected by geometric gate (overlap)")
            return False

        # Step 4: 20-step FIRE relaxation (per original paper).
        # Both foundation and fine-tuned models agree: O at crowded sites
        # naturally detaches (d_Ir → 2.9). This is physical — the bonding
        # gate below catches and rejects these moves.
        try:
            self._relax_adsorbates(replica)
            replica.relaxed_atoms = replica.real_atoms.copy()
            replica.save_state("after")
        except Exception as exc:
            self.logger.warning("Local relax failed (%s), using unrelaxed geometry", exc)

        # Step 4b: bonding gate — reject if any adsorbate O detached
        # during relaxation (both models push unbondable O to 2.9+).
        if not self._bonding_ok(replica.real_atoms):
            event.backward()
            self.logger.debug("MC move rejected: adsorbate detached after relax")
            return False

        # Step 5: Metropolis on (relaxed) energies. The criterion internally
        # restores "before", reads cached energy, then restores "after" and
        # evaluates the new energy. If it rejects it does not roll back for
        # us, so we backward() manually.
        accept = event.criterion(replica)
        if not accept:
            event.backward()

        return bool(accept)

    def _mc_step_mtm(self, replica_idx: int) -> bool:
        """Perform one Multiple-Try Metropolis (MTM) step on a replica.

        References:
            Liu, Liang, Wong. "The multiple-try method and local optimization
            in Metropolis sampling." J. Am. Stat. Assoc. 95 (2000) 121.

        Algorithm (Liu 2000 Eq 2.4 — independent-proposal MTM, approximate
        for our conditional ChangeProposal but tight in the large-state-space
        limit that HRE-MC operates in):

            1. At current state x with energy E_x:
            2. Sample k trial actions a_1, ..., a_k from ChangeProposal (k = n_trials).
            3. For each trial i: forward → geometry_ok → FIRE relax → bonding_ok →
               MACE energy E_i. Invalid trials are dropped. Roll back state.
            4. If no trials are valid, reject (stay at x).
            5. Compute Boltzmann weights w_i = exp(-β (E_i - E_ref))
               and w_x = exp(-β (E_x - E_ref)) with E_ref = min(E_x, min_i E_i)
               for numerical stability.
            6. Pick trial j with probability p_j = w_j / Σ_i w_i.
            7. Accept with probability α = min(1, W_forward / W_reverse) where
               W_forward = Σ_i w_i  and  W_reverse = W_forward - w_j + w_x
               (the chosen y_j is swapped out for x in the reverse-direction
               sum). This is Liu 2000 Eq 2.4 (independent proposals); the
               strictly unbiased Eq 2.6 for *conditional* symmetric proposals
               would require resampling k−1 reverse trials from y_j (adding
               another k−1 FIRE + MACE evaluations per step). Since
               ChangeProposal is "effectively independent" in our large
               (~10^18) slab configuration space — two independent trials
               almost never collide on the same y — the Eq 2.4 approximation
               is tight in practice. The shortcut fails only in pathological
               small state spaces (e.g. a 2-state toy) where q(y|x) is
               nearly deterministic; unit tests for the formula itself use
               k=1 (equivalent to plain Metropolis) and are verified in
               tests/test_mtm_math.py.
            8. If accepted, re-apply the picked action via FixedActionChangeProposal
               and re-run FIRE. (The ~5% extra cost per accept is negligible
               compared to the k trial relaxes already spent.)

        Cost per MC step: k forward/backward cycles + k FIRE runs + k MACE
        energies, plus 1 extra FIRE + MACE for each accepted move. For k=4
        the per-step cost is ~4.2× single Metropolis, but the effective
        sample length increases by a similar factor near phase boundaries
        where single-MC gets stuck.

        Only supports semigrand mode (ChangeProposal). Canonical
        (SwitchProposal) is not yet implemented; the dispatch in _mc_step
        falls back to single-try for that case.
        """
        replica = self.replicas[replica_idx]
        temp = self.conditions[replica_idx].temperature
        beta = 1.0 / temp
        k = self.n_trials

        E_x = self._get_pourbaix_potential(replica_idx)

        # Generate k trial candidates. Each trial goes through the full
        # forward → gate → relax → gate → energy pipeline, is recorded,
        # then rolled back so the next trial starts from x.
        trials: list[dict] = []
        for trial_i in range(k):
            # Sample a fresh action from the unconditional ChangeProposal
            sampler = ChangeProposal(
                system=replica, adsorbate_list=self.adsorbates.copy()
            )
            action = sampler.get_action()

            # Apply via fixed-action proposal so event.forward/backward
            # see a consistent, repeatable action.
            fixed_prop = FixedActionChangeProposal(
                system=replica,
                adsorbate_list=self.adsorbates.copy(),
                fixed_action=action,
            )
            event = Change(replica, fixed_prop, MetropolisCriterion(temp))
            event.forward()

            valid = self._geometry_ok(replica.real_atoms)
            if valid:
                try:
                    self._relax_adsorbates(replica)
                    replica.relaxed_atoms = replica.real_atoms.copy()
                    replica.save_state("after")
                    valid = self._bonding_ok(replica.real_atoms)
                except Exception as exc:
                    self.logger.debug(
                        "MTM trial %d relax failed: %s", trial_i, exc
                    )
                    valid = False

            if valid:
                energy = float(replica.get_surface_energy())
                trials.append(
                    {"action": action, "energy": energy, "trial_i": trial_i}
                )
            else:
                self.logger.debug(
                    "MTM trial %d rejected at gate (geometry or bonding)",
                    trial_i,
                )

            event.backward()

        if not trials:
            self.logger.debug("MTM step: all %d trials failed gates", k)
            return False

        # Boltzmann-weighted pick. Shift energies by E_ref before exp() to
        # avoid overflow/underflow when (E_x, E_i) are both large and close.
        energies = np.array([t["energy"] for t in trials], dtype=np.float64)
        E_ref = float(min(energies.min(), E_x))
        w_y = np.exp(-beta * (energies - E_ref))
        w_x = float(np.exp(-beta * (E_x - E_ref)))
        p_pick = w_y / w_y.sum()
        j = int(np.random.choice(len(trials), p=p_pick))

        # MTM acceptance: Liu 2000 Eq. 2.6 for symmetric proposals.
        # α = min(1, W_forward / W_reverse)
        #   W_forward = Σ_i w(y_i)             (all trial weights)
        #   W_reverse = Σ_i w(x_i) with x_j = x (swap the picked y_j for x)
        #             = W_forward - w_y[j] + w_x
        W_forward = float(w_y.sum())
        W_reverse = W_forward - float(w_y[j]) + w_x
        alpha = min(1.0, W_forward / W_reverse)
        u = float(np.random.random())
        self.logger.debug(
            "MTM: k=%d valid=%d picked=%d α=%.4f u=%.4f E_x=%.4f E_j=%.4f",
            k, len(trials), j, alpha, u, E_x, energies[j],
        )

        if u >= alpha:
            # Rejected. Replica is already back at x from the last event.backward().
            return False

        # Accepted — re-apply the chosen action and re-run FIRE so the
        # final stored geometry matches the one whose energy we committed
        # to.
        chosen = trials[j]
        fixed_prop = FixedActionChangeProposal(
            system=replica,
            adsorbate_list=self.adsorbates.copy(),
            fixed_action=chosen["action"],
        )
        event = Change(replica, fixed_prop, MetropolisCriterion(temp))
        event.forward()
        try:
            self._relax_adsorbates(replica)
            replica.relaxed_atoms = replica.real_atoms.copy()
            replica.save_state("after")
        except Exception as exc:
            self.logger.warning("MTM accepted-trial re-relax failed: %s", exc)

        # Cache the chosen trial's energy so downstream _get_pourbaix_potential
        # reads the same value we used in the acceptance decision. FIRE is
        # nominally deterministic but momentum init may differ slightly.
        try:
            replica.results["surface_energy"] = chosen["energy"]
        except AttributeError:
            pass

        return True

    # ------------------------------------------------------------------
    # Helpers: geometric gate + local adsorbate relaxation
    # ------------------------------------------------------------------

    @staticmethod
    def _geometry_ok(atoms, min_pair: float = _MC_MIN_PAIR_DIST) -> bool:
        """Reject obvious collisions before a MACE eval."""
        if len(atoms) < 2:
            return True
        d = atoms.get_all_distances(mic=True)
        # mask the diagonal
        np.fill_diagonal(d, np.inf)
        return bool(d.min() >= min_pair)

    @staticmethod
    def _bonding_ok(atoms, max_ir_o: float = 2.2, max_o_h: float = 1.3,
                    max_o_o: float = 1.6, z_base_top: float | None = None) -> bool:
        """Reject if any adsorbate atom is not properly bonded.

        Every adsorbate O must have at least one Ir within max_ir_o,
        OR at least one other O within max_o_o (O-O peroxo bond —
        a legitimate OER intermediate, not a floating atom).
        Every H must have at least one O within max_o_h.
        """
        syms = atoms.get_chemical_symbols()
        z = atoms.positions[:, 2]
        if z_base_top is None:
            ir_z = [z[i] for i, s in enumerate(syms) if s == "Ir"]
            z_base_top = max(ir_z) if ir_z else z.max()
        ir_idx = [i for i, s in enumerate(syms) if s == "Ir"]
        o_idx = [i for i, s in enumerate(syms) if s == "O"]
        for i, s in enumerate(syms):
            if s == "O" and z[i] > z_base_top + 0.3:
                d_ir = atoms.get_distances(i, ir_idx, mic=True)
                if d_ir.min() > max_ir_o:
                    # Not directly bonded to Ir — check for surface peroxo:
                    # this O must bond to another O (< 1.6 Å) that IS
                    # bonded to Ir (< max_ir_o). Both O floating = reject.
                    other_o = [j for j in o_idx if j != i]
                    is_peroxo = False
                    if other_o:
                        d_oo = atoms.get_distances(i, other_o, mic=True)
                        for k, d in enumerate(d_oo):
                            if d < max_o_o:
                                # This O bonds to other_o[k] — check that one bonds to Ir
                                partner = other_o[k]
                                d_partner_ir = atoms.get_distances(partner, ir_idx, mic=True)
                                if d_partner_ir.min() <= max_ir_o:
                                    is_peroxo = True
                                    break
                    if not is_peroxo:
                        return False
            elif s == "H":
                d = atoms.get_distances(i, o_idx, mic=True)
                if d.min() > max_o_h:
                    return False
        return True

    def _relax_adsorbates(
        self,
        replica: SurfaceSystem,
        fmax: float = _MC_RELAX_FMAX,
        steps: int = _MC_RELAX_STEPS,
    ) -> None:
        """Run a short FIRE relaxation on adsorbate atoms only.

        We freeze every atom that is part of the pristine slab and let only
        atoms introduced via MC moves (tracked in the `ads_group` array) move.
        This is much cheaper than a full slab relax and is all we need to
        bring a newly-placed O/OH onto its bonding distance.
        """
        atoms = replica.real_atoms
        if atoms.calc is None:
            replica.set_calc(replica.calc)  # rebind
            atoms = replica.real_atoms

        # Identify adsorbate atoms via `ads_group` (non-zero = adsorbate)
        try:
            ads_group = atoms.get_array("ads_group")
            ads_idx = np.where(ads_group != 0)[0].tolist()
        except (KeyError, RuntimeError):
            ads_idx = []

        if not ads_idx:
            return  # nothing moved; skip relax entirely

        frozen_idx = [i for i in range(len(atoms)) if i not in ads_idx]
        # Preserve any user-supplied FixAtoms and add a second constraint for
        # this transient relax.
        orig_constraints = atoms.constraints
        atoms.set_constraint(list(orig_constraints) + [FixAtoms(indices=frozen_idx)])
        try:
            dyn = FIRE(atoms, logfile=None)
            dyn.run(fmax=fmax, steps=steps)
        finally:
            atoms.set_constraint(orig_constraints)

    def _sweep(self, replica_idx: int, sweep_size: int) -> dict:
        """Run one sweep on a single replica."""
        n_accept = 0
        for _ in range(sweep_size):
            n_accept += self._mc_step(replica_idx)

        energy = self.replicas[replica_idx].get_surface_energy()

        # Check uncertainty if tracker is active
        if self.uncertainty_tracker:
            self.uncertainty_tracker.check(self.replicas[replica_idx])

        return {
            "energy": float(energy),
            "acceptance_rate": n_accept / sweep_size,
            "adsorption_count": self.replicas[replica_idx].num_adsorbates,
        }

    def _get_pourbaix_potential(self, replica_idx: int) -> float:
        """Get cached Pourbaix potential for a replica."""
        try:
            return float(self.replicas[replica_idx].results["surface_energy"])
        except (KeyError, AttributeError):
            return float(self.replicas[replica_idx].get_surface_energy(recalculate=True))

    def _get_slab_energy(self, replica_idx: int) -> float:
        """Get the MACE potential energy of a replica's slab (for ΔG₁ computation)."""
        replica = self.replicas[replica_idx]
        calc = replica.calc
        atoms = replica.unrelaxed_atoms if not replica.relax_atoms else (
            replica.relaxed_atoms or replica.unrelaxed_atoms
        )
        return calc.get_potential_energy(atoms=atoms)

    def _compute_cross_potential(self, config_idx: int, condition_idx: int) -> float:
        """Compute Pourbaix potential of config from replica `config_idx` evaluated
        under conditions of replica `condition_idx`.

        This is the key operation for Hamiltonian exchange. Since DeltaG1 is independent
        of (pH, phi), we only recompute DeltaG2 analytically. No MACE evaluation needed.

        Args:
            config_idx: Index of the replica whose configuration we evaluate.
            condition_idx: Index of the replica whose conditions we use.

        Returns:
            float: Pourbaix potential Omega = -(DeltaG1 + DeltaG2) under target conditions.
        """
        replica = self.replicas[config_idx]
        atoms = replica.relaxed_atoms if replica.relaxed_atoms else replica.unrelaxed_atoms
        symbols = atoms.get_chemical_symbols()

        # DeltaG1: independent of (pH, phi), use any replica's pourbaix_atoms for std energies.
        # Note: atom_std_state_energy is from PhaseDiagram, independent of (pH, phi).
        # It's safe to use config_idx's pourbaix_atoms here because this value
        # is the same across all replicas.
        slab_energy = self._get_slab_energy(config_idx)
        dG1 = compute_dG1(symbols, self.replica_pourbaix_atoms[config_idx], slab_energy)

        # DeltaG2: recompute under target (pH, phi) conditions.
        # Use electrochemical_temp (298K) for Nernst equation, NOT MC temperature.
        cond = self.conditions[condition_idx]
        dG2 = compute_dG2(
            symbols, self.replica_pourbaix_atoms[condition_idx],
            pH=cond.pH, phi=cond.phi, temperature=cond.electrochemical_temp,
        )

        return -(dG1 + dG2)

    def _attempt_swap(self, i: int, j: int) -> bool:
        """Attempt Hamiltonian replica exchange between replicas i and j.

        Acceptance criterion:
            Delta = beta_i * Omega_i(x_j) + beta_j * Omega_j(x_i)
                  - beta_i * Omega_i(x_i) - beta_j * Omega_j(x_j)
            P(swap) = min(1, exp(-Delta))

        where beta = 1/kT_mc is the inverse MC sampling temperature.
        If Delta <= 0, the swap lowers the combined "energy" and is always accepted.
        If Delta > 0, it is accepted with probability exp(-Delta).

        For replicas at the same MC temperature but different (pH, phi):
            The DeltaG1 terms cancel, and Delta depends only on DeltaG2 differences
            (analytical, no MACE evaluation needed).

        Returns:
            bool: Whether the swap was accepted.
        """
        beta_i = 1.0 / self.conditions[i].temperature
        beta_j = 1.0 / self.conditions[j].temperature

        # Current energies (cached)
        Omega_i_xi = self._get_pourbaix_potential(i)
        Omega_j_xj = self._get_pourbaix_potential(j)

        # Cross energies: evaluate each config under the other's conditions
        Omega_i_xj = self._compute_cross_potential(config_idx=j, condition_idx=i)
        Omega_j_xi = self._compute_cross_potential(config_idx=i, condition_idx=j)

        delta = (beta_i * Omega_i_xj + beta_j * Omega_j_xi
                 - beta_i * Omega_i_xi - beta_j * Omega_j_xj)

        if delta <= 0:
            accept = True
        else:
            try:
                accept = np.random.rand() < np.exp(-delta)
            except OverflowError:
                accept = False

        pair = (min(i, j), max(i, j))
        self._swap_stats[pair].attempted += 1

        if accept:
            self._swap_stats[pair].accepted += 1
            # Swap configurations (surface systems), keep conditions fixed
            self.replicas[i], self.replicas[j] = self.replicas[j], self.replicas[i]
            # Re-attach correct calculators after swap
            calc_i, calc_j = self.replicas[i].calc, self.replicas[j].calc
            self.replicas[i].set_calc(calc_j)
            self.replicas[j].set_calc(calc_i)
            self.logger.debug("Swap %d↔%d accepted (Δ=%.3f)", i, j, delta)
        else:
            self.logger.debug("Swap %d↔%d rejected (Δ=%.3f)", i, j, delta)

        return accept

    def _swap_round(self, sweep_num: int) -> int:
        """Attempt swaps along a randomly chosen dimension."""
        n_accepted = 0
        # Shuffle pairs to avoid bias
        pairs = self._swap_pairs.copy()
        np.random.shuffle(pairs)

        # Alternate which pairs to attempt (like even/odd in standard PT)
        parity = sweep_num % 2
        selected = pairs[parity::2] if len(pairs) > 1 else pairs

        for i, j in selected:
            n_accepted += self._attempt_swap(i, j)

        return n_accepted

    def run(
        self,
        surface: SurfaceSystem,
        total_sweeps: int = 100,
        sweep_size: int = 20,
        save_interval: int = 5,
        run_folder: str | Path | None = None,
        logger: logging.Logger | None = None,
        **kwargs,
    ) -> dict:
        """Run Hamiltonian Replica Exchange MC.

        Args:
            surface: Initial SurfaceSystem (copied for each replica).
            total_sweeps: Number of MC sweeps per replica.
            sweep_size: Steps per sweep.
            save_interval: Save structure snapshots every N sweeps (default 5).
                Publication-quality sampling typically wants 5 (20 snapshots
                per replica over 100 sweeps → 140 for 7 replicas → plenty
                for k-means clustering + Pourbaix phase statistics).
                Set to total_sweeps to only save the final state.
            run_folder: Output directory.

        Returns:
            dict: Per-replica histories, energies, swap statistics, and diagram data.
        """
        if logger:
            self.logger = logger

        run_folder = Path(run_folder) if run_folder else Path("hre_run")
        run_folder.mkdir(parents=True, exist_ok=True)

        # Create replica directories
        replica_folders = []
        for i, cond in enumerate(self.conditions):
            rf = run_folder / f"replica_{i}_pH{cond.pH}_phi{cond.phi}_T{cond.temperature:.4f}"
            rf.mkdir(parents=True, exist_ok=True)
            replica_folders.append(rf)

        self.replicas = self._create_replicas(surface)

        self.logger.info(
            "Starting Hamiltonian RE: %d replicas, %d sweeps, %d steps/sweep",
            self.n_replicas, total_sweeps, sweep_size,
        )
        for i, cond in enumerate(self.conditions):
            self.logger.info("  Replica %d: %s", i, cond)

        # Results
        results = {
            "conditions": [
                {"temperature": c.temperature, "pH": c.pH, "phi": c.phi}
                for c in self.conditions
            ],
            "replica_energies": [[] for _ in range(self.n_replicas)],
            "replica_accept_rates": [[] for _ in range(self.n_replicas)],
            "replica_ads_counts": [[] for _ in range(self.n_replicas)],
            "replica_histories": [[] for _ in range(self.n_replicas)],
            "swap_stats": {},
            "uncertainty_flags": 0,
        }

        for sweep_num in range(total_sweeps):
            self.logger.info("Sweep %d / %d", sweep_num + 1, total_sweeps)

            # Independent MC sweeps on all replicas
            for i in range(self.n_replicas):
                sweep_result = self._sweep(i, sweep_size)
                results["replica_energies"][i].append(sweep_result["energy"])
                results["replica_accept_rates"][i].append(sweep_result["acceptance_rate"])
                results["replica_ads_counts"][i].append(sweep_result["adsorption_count"])

                snapshot = self.replicas[i].copy(copy_calc=False)
                snapshot.unset_calc()
                results["replica_histories"][i].append(snapshot)

            # Hamiltonian replica exchange
            if (sweep_num + 1) % self.swap_interval == 0:
                n_accepted = self._swap_round(sweep_num)
                self.logger.info(
                    "Swap round: %d accepted. Per-pair rates: %s",
                    n_accepted,
                    {f"{i}↔{j}": f"{s.rate:.0%}" for (i, j), s in self._swap_stats.items()},
                )

            # Save structures periodically
            if (sweep_num + 1) % save_interval == 0 or sweep_num == total_sweeps - 1:
                for i in range(self.n_replicas):
                    self.replicas[i].save_structures(
                        sweep_num=sweep_num + 1, save_folder=replica_folders[i],
                    )

        # Record final swap stats
        results["swap_stats"] = {
            f"{i}↔{j}": {"attempted": s.attempted, "accepted": s.accepted, "rate": s.rate}
            for (i, j), s in self._swap_stats.items()
        }

        if self.uncertainty_tracker:
            results["uncertainty_flags"] = self.uncertainty_tracker.total_flagged

        self._save_results(results, run_folder)
        return results

    def _save_results(self, results: dict, run_folder: Path) -> None:
        """Save results to disk."""
        import pandas as pd

        for i in range(self.n_replicas):
            cond = self.conditions[i]
            df = pd.DataFrame({
                "energy": results["replica_energies"][i],
                "accept_rate": results["replica_accept_rates"][i],
                "ads_count": results["replica_ads_counts"][i],
            })
            df.to_csv(
                run_folder / f"replica_{i}_pH{cond.pH}_phi{cond.phi}_stats.csv",
                index=False, float_format="%.4f",
            )

        # Save diagram data: for each (pH, φ) point, the equilibrium structure
        diagram_data = {}
        for i, cond in enumerate(self.conditions):
            key = f"pH{cond.pH}_phi{cond.phi}"
            energies = results["replica_energies"][i]
            if energies:
                diagram_data[key] = {
                    "condition": {"pH": cond.pH, "phi": cond.phi, "T": cond.temperature},
                    "final_energy": energies[-1],
                    "final_ads_count": results["replica_ads_counts"][i][-1],
                    "mean_accept_rate": float(np.mean(results["replica_accept_rates"][i])),
                }

        import json
        with open(run_folder / "pourbaix_diagram_data.json", "w") as f:
            json.dump(diagram_data, f, indent=2)

        # Save swap stats
        with open(run_folder / "swap_stats.json", "w") as f:
            json.dump(results["swap_stats"], f, indent=2)

        self.logger.info("Results saved to %s", run_folder)

    def get_pourbaix_diagram_data(self, results: dict) -> dict:
        """Extract data for plotting a surface Pourbaix diagram.

        Returns:
            dict: {(pH, phi): {"energy": ..., "ads_count": ..., "composition": ...}}
        """
        diagram = {}
        for i, cond in enumerate(self.conditions):
            energies = results["replica_energies"][i]
            ads_counts = results["replica_ads_counts"][i]
            histories = results["replica_histories"][i]

            if histories:
                final = histories[-1]
                composition = dict(Counter(final.real_atoms.get_chemical_symbols()))
            else:
                composition = {}

            diagram[(cond.pH, cond.phi)] = {
                "energy": energies[-1] if energies else None,
                "ads_count": ads_counts[-1] if ads_counts else None,
                "composition": composition,
                "temperature": cond.temperature,
            }
        return diagram
