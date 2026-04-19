"""Parallel Tempering (Replica Exchange) Monte Carlo for surface Pourbaix sampling.

Runs multiple replicas at different temperatures simultaneously. After each sweep,
adjacent replicas attempt configuration swaps with the detailed-balance criterion:

    P(swap) = min(1, exp((1/kT_i - 1/kT_j) * (E_i - E_j)))

This allows high-temperature replicas to explore broadly while low-temperature
replicas sample the equilibrium distribution accurately.

Includes a background monitoring agent (PTMonitor) that continuously checks
replica health and triggers corrective actions:
  - NaN/Inf energy → reset replica from nearest healthy neighbor
  - Zero acceptance rate → insert intermediate temperature
  - Swap rate too low → refine temperature ladder
  - Energy divergence → restart replica from checkpoint
"""

import copy
import logging
import pickle
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mcmc.events.criterion import MetropolisCriterion
from mcmc.events.event import Change, Exchange
from mcmc.events.proposal import ChangeProposal, SwitchProposal
from mcmc.system import SurfaceSystem


# ──────────────────────────────────────────────────────────────
#  Background Monitor Agent
# ──────────────────────────────────────────────────────────────

@dataclass
class HealthStatus:
    """Health status of a single replica."""
    replica_idx: int
    is_healthy: bool = True
    issue: str = ""
    action: str = ""


@dataclass
class MonitorConfig:
    """Configuration for the PT monitor agent.

    Attributes:
        check_interval: Check health every N sweeps.
        max_energy: Absolute energy threshold — flag if |E| exceeds this.
        min_accept_rate: Flag replica if rolling acceptance rate drops below this.
        accept_rate_window: Number of recent sweeps to average for acceptance rate.
        min_swap_rate: Flag if cumulative swap acceptance drops below this.
        max_energy_jump: Flag if energy changes by more than this between sweeps.
        max_consecutive_resets: Abort run if a replica resets more than this many times.
        auto_refine_temps: If True, insert intermediate temperatures when swap rate is low.
    """
    check_interval: int = 1
    max_energy: float = 1e4
    min_accept_rate: float = 0.01
    accept_rate_window: int = 5
    min_swap_rate: float = 0.05
    max_energy_jump: float = 100.0
    max_consecutive_resets: int = 5
    auto_refine_temps: bool = False


class PTMonitor:
    """Background monitoring agent for Parallel Tempering MC.

    Runs as a daemon thread that periodically inspects shared state (energies,
    acceptance rates, swap rates) posted by the main MC loop. When it detects
    problems, it sets action flags that the main loop reads and executes.

    Thread safety: the monitor only *reads* shared diagnostics and *writes*
    to an action queue. The main loop is the only writer of diagnostics and
    the only executor of actions.
    """

    def __init__(
        self,
        config: MonitorConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config or MonitorConfig()
        self.logger = logger or logging.getLogger(__name__ + ".monitor")

        # Shared state — written by main loop, read by monitor
        self._lock = threading.Lock()
        self._diagnostics: dict = {
            "energies": [],        # list of lists: [replica][sweep]
            "accept_rates": [],    # list of lists: [replica][sweep]
            "swap_rates": [],      # list of floats per swap round
            "sweep_num": 0,
            "n_replicas": 0,
            "temperatures": [],
        }

        # Action queue — written by monitor, read/consumed by main loop
        self._actions: list[dict] = []
        self._action_lock = threading.Lock()

        # Reset counters per replica
        self._reset_counts: dict[int, int] = {}

        # Control
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self, n_replicas: int, temperatures: list[float]) -> None:
        """Start the monitor daemon thread."""
        self._diagnostics["n_replicas"] = n_replicas
        self._diagnostics["temperatures"] = temperatures
        self._diagnostics["energies"] = [[] for _ in range(n_replicas)]
        self._diagnostics["accept_rates"] = [[] for _ in range(n_replicas)]
        self._reset_counts = {i: 0 for i in range(n_replicas)}
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.logger.info("Monitor agent started (check every %d sweeps)", self.config.check_interval)

    def stop(self) -> None:
        """Stop the monitor thread."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.logger.info("Monitor agent stopped")

    def post_sweep(
        self,
        sweep_num: int,
        energies: list[float],
        accept_rates: list[float],
        swap_rate: float | None = None,
    ) -> None:
        """Called by main loop after each sweep to update diagnostics."""
        with self._lock:
            self._diagnostics["sweep_num"] = sweep_num
            for i, (e, ar) in enumerate(zip(energies, accept_rates)):
                self._diagnostics["energies"][i].append(e)
                self._diagnostics["accept_rates"][i].append(ar)
            if swap_rate is not None:
                self._diagnostics["swap_rates"].append(swap_rate)

    def get_actions(self) -> list[dict]:
        """Called by main loop to consume pending actions."""
        with self._action_lock:
            actions = self._actions.copy()
            self._actions.clear()
        return actions

    def _enqueue_action(self, action: dict) -> None:
        with self._action_lock:
            self._actions.append(action)

    def _monitor_loop(self) -> None:
        """Background loop: periodically check health and enqueue corrective actions."""
        while self._running:
            time.sleep(0.1)  # Lightweight polling — actual checks gated by sweep count

            with self._lock:
                sweep_num = self._diagnostics["sweep_num"]
                n_replicas = self._diagnostics["n_replicas"]

            if sweep_num == 0 or n_replicas == 0:
                continue
            if sweep_num % self.config.check_interval != 0:
                continue

            statuses = self._check_all_replicas()

            for status in statuses:
                if not status.is_healthy:
                    self.logger.warning(
                        "Replica %d UNHEALTHY: %s → action: %s",
                        status.replica_idx, status.issue, status.action,
                    )
                    self._enqueue_action({
                        "type": status.action,
                        "replica_idx": status.replica_idx,
                        "reason": status.issue,
                        "sweep_num": sweep_num,
                    })

            # Check swap health
            self._check_swap_health()

    def _check_all_replicas(self) -> list[HealthStatus]:
        """Run all health checks on every replica."""
        statuses = []
        with self._lock:
            n_replicas = self._diagnostics["n_replicas"]
            for i in range(n_replicas):
                status = self._check_replica(i)
                statuses.append(status)
        return statuses

    def _check_replica(self, idx: int) -> HealthStatus:
        """Check health of a single replica. Must be called with self._lock held."""
        energies = self._diagnostics["energies"][idx]
        accept_rates = self._diagnostics["accept_rates"][idx]

        if not energies:
            return HealthStatus(replica_idx=idx)

        latest_E = energies[-1]

        # Check 1: NaN or Inf energy
        if np.isnan(latest_E) or np.isinf(latest_E):
            self._reset_counts[idx] += 1
            if self._reset_counts[idx] > self.config.max_consecutive_resets:
                return HealthStatus(idx, False, "NaN/Inf energy, max resets exceeded", "abort")
            return HealthStatus(idx, False, f"Energy is {latest_E}", "reset_from_neighbor")

        # Check 2: Energy too large
        if abs(latest_E) > self.config.max_energy:
            self._reset_counts[idx] += 1
            if self._reset_counts[idx] > self.config.max_consecutive_resets:
                return HealthStatus(idx, False, "Energy diverged, max resets exceeded", "abort")
            return HealthStatus(
                idx, False, f"|E|={abs(latest_E):.1f} > {self.config.max_energy}", "reset_from_neighbor"
            )

        # Check 3: Energy jump between consecutive sweeps
        if len(energies) >= 2:
            jump = abs(energies[-1] - energies[-2])
            if jump > self.config.max_energy_jump:
                return HealthStatus(
                    idx, False,
                    f"Energy jump {jump:.1f} > {self.config.max_energy_jump}",
                    "reset_from_checkpoint",
                )

        # Check 4: Acceptance rate too low
        window = self.config.accept_rate_window
        recent_rates = accept_rates[-window:]
        mean_rate = np.mean(recent_rates)
        if len(recent_rates) >= window and mean_rate < self.config.min_accept_rate:
            return HealthStatus(
                idx, False,
                f"Accept rate {mean_rate:.3f} < {self.config.min_accept_rate} over {window} sweeps",
                "warn_stuck",
            )

        # Healthy — reset the consecutive reset counter
        self._reset_counts[idx] = 0
        return HealthStatus(replica_idx=idx)

    def _check_swap_health(self) -> None:
        """Check if swap acceptance rate is too low."""
        with self._lock:
            swap_rates = self._diagnostics["swap_rates"]

        if len(swap_rates) < 5:
            return

        recent_swap_rate = np.mean(swap_rates[-5:])
        if recent_swap_rate < self.config.min_swap_rate:
            self.logger.warning(
                "Swap rate %.3f < %.3f — temperature ladder may need refinement",
                recent_swap_rate, self.config.min_swap_rate,
            )
            if self.config.auto_refine_temps:
                self._enqueue_action({
                    "type": "refine_temperatures",
                    "reason": f"swap rate {recent_swap_rate:.3f} too low",
                })


class ParallelTemperingMC:
    """Parallel Tempering (Replica Exchange) MC for surface reconstruction sampling.

    Each replica is a SurfaceSystem at a different temperature. Replicas run
    independent MC sweeps, then attempt pairwise swaps between neighbors in
    temperature space.

    Args:
        temperatures: List of temperatures (in kT, eV) for replicas, sorted ascending.
        adsorbates: List of adsorbate species (e.g. ["Cu", "Sn", "O", "HO"]).
        canonical: If True, use canonical (swap) moves; otherwise semi-grand canonical.
        num_ads_atoms: Number of adsorbate atoms (required if canonical=True).
        swap_interval: Attempt replica swaps every N sweeps.
        swap_scheme: "sequential" (even/odd alternation) or "random" (random pair).
    """

    def __init__(
        self,
        temperatures: list[float],
        adsorbates: list[str] | None = None,
        canonical: bool = False,
        num_ads_atoms: int = 0,
        swap_interval: int = 1,
        swap_scheme: str = "sequential",
        monitor_config: MonitorConfig | None = None,
        **kwargs,
    ) -> None:
        if len(temperatures) < 2:
            raise ValueError("Need at least 2 temperatures for parallel tempering.")
        self.temperatures = sorted(temperatures)
        self.n_replicas = len(self.temperatures)
        self.adsorbates = adsorbates or []
        self.canonical = canonical
        self.num_ads_atoms = num_ads_atoms
        self.swap_interval = swap_interval
        self.swap_scheme = swap_scheme
        self.monitor_config = monitor_config or MonitorConfig()

        self.replicas: list[SurfaceSystem] = []
        self._checkpoints: dict[int, SurfaceSystem] = {}  # replica_idx -> last good state
        self._initial_surface: SurfaceSystem | None = None
        self.logger = kwargs.get("logger", logging.getLogger(__name__))

    def _create_replicas(self, surface: SurfaceSystem) -> list[SurfaceSystem]:
        """Create independent surface copies for each temperature replica.

        All replicas start from the same initial configuration but will diverge
        as MC evolves them at different temperatures.
        """
        replicas = []
        for i in range(self.n_replicas):
            replica = surface.copy(copy_calc=False)
            # Share the same calculator (MACE model) to save GPU memory
            replica.set_calc(surface.calc)
            replicas.append(replica)
            self.logger.info(
                "Replica %d: T = %.4f kT", i, self.temperatures[i]
            )
        return replicas

    def _mc_step(self, replica: SurfaceSystem, temperature: float) -> bool:
        """Perform a single MC step on a replica at given temperature.

        Returns:
            bool: Whether the proposed move was accepted.
        """
        if self.canonical:
            proposal = SwitchProposal(
                system=replica, adsorbate_list=self.adsorbates.copy()
            )
            event = Exchange(replica, proposal, MetropolisCriterion(temperature))
        else:
            proposal = ChangeProposal(
                system=replica, adsorbate_list=self.adsorbates.copy()
            )
            event = Change(replica, proposal, MetropolisCriterion(temperature))

        accept, replica = event.acceptance()
        return accept

    def _sweep(self, replica: SurfaceSystem, temperature: float, sweep_size: int) -> dict:
        """Perform one sweep (sweep_size MC steps) on a single replica.

        Returns:
            dict with keys: energy, acceptance_rate, adsorption_count.
        """
        n_accept = 0
        for _ in range(sweep_size):
            n_accept += self._mc_step(replica, temperature)

        energy = replica.get_surface_energy()
        return {
            "energy": float(energy),
            "acceptance_rate": n_accept / sweep_size,
            "adsorption_count": replica.num_adsorbates,
        }

    def _get_replica_energy(self, replica: SurfaceSystem) -> float:
        """Get cached surface energy for a replica (no recalculation)."""
        try:
            return float(replica.results["surface_energy"])
        except KeyError:
            return float(replica.get_surface_energy(recalculate=True))

    def _attempt_swap(self, i: int, j: int) -> bool:
        """Attempt to swap configurations between replicas i and j.

        Uses the replica exchange criterion:
            P(swap) = min(1, exp((beta_i - beta_j) * (E_i - E_j)))

        where beta = 1/kT.

        Returns:
            bool: Whether the swap was accepted.
        """
        E_i = self._get_replica_energy(self.replicas[i])
        E_j = self._get_replica_energy(self.replicas[j])
        beta_i = 1.0 / self.temperatures[i]
        beta_j = 1.0 / self.temperatures[j]

        delta = (beta_i - beta_j) * (E_i - E_j)

        if delta <= 0:
            accept = True
        else:
            accept = np.random.rand() < np.exp(-delta)

        if accept:
            # Swap the surface configurations (not the temperatures)
            self.replicas[i], self.replicas[j] = self.replicas[j], self.replicas[i]
            self.logger.debug(
                "Swap accepted: replicas %d <-> %d (E=%.3f, %.3f; T=%.4f, %.4f)",
                i, j, E_i, E_j, self.temperatures[i], self.temperatures[j],
            )
        else:
            self.logger.debug(
                "Swap rejected: replicas %d <-> %d (delta=%.3f)", i, j, delta
            )

        return accept

    def _swap_round(self, sweep_num: int) -> int:
        """Perform one round of replica swap attempts.

        For 'sequential' scheme: alternate between even pairs (0-1, 2-3, ...)
        and odd pairs (1-2, 3-4, ...) on successive rounds.

        For 'random' scheme: pick a random adjacent pair.

        Returns:
            int: Number of accepted swaps.
        """
        n_accepted = 0

        if self.swap_scheme == "sequential":
            # Alternate even/odd pairs to satisfy detailed balance
            parity = sweep_num % 2
            for k in range(parity, self.n_replicas - 1, 2):
                n_accepted += self._attempt_swap(k, k + 1)

        elif self.swap_scheme == "random":
            k = np.random.randint(0, self.n_replicas - 1)
            n_accepted += self._attempt_swap(k, k + 1)

        return n_accepted

    def _handle_actions(self, actions: list[dict], run_folder: Path) -> bool:
        """Process corrective actions from the monitor.

        Returns:
            bool: True if simulation should abort.
        """
        for action in actions:
            atype = action["type"]
            idx = action.get("replica_idx")
            reason = action.get("reason", "")

            if atype == "abort":
                self.logger.error(
                    "ABORT requested for replica %s: %s", idx, reason
                )
                # Save crash dump before aborting
                self._save_crash_dump(run_folder, action)
                return True

            elif atype == "reset_from_neighbor":
                self._reset_replica_from_neighbor(idx)
                self.logger.warning(
                    "Replica %d reset from nearest healthy neighbor (reason: %s)",
                    idx, reason,
                )

            elif atype == "reset_from_checkpoint":
                self._reset_replica_from_checkpoint(idx)
                self.logger.warning(
                    "Replica %d reset from checkpoint (reason: %s)", idx, reason
                )

            elif atype == "warn_stuck":
                self.logger.warning(
                    "Replica %d appears stuck: %s (no auto-fix, continuing)",
                    idx, reason,
                )

            elif atype == "refine_temperatures":
                self.logger.warning(
                    "Temperature refinement requested: %s (not yet implemented)",
                    reason,
                )

        return False

    def _reset_replica_from_neighbor(self, idx: int) -> None:
        """Reset a broken replica by copying config from the nearest healthy neighbor."""
        # Find nearest healthy neighbor (prefer lower temperature)
        for offset in range(1, self.n_replicas):
            for neighbor in [idx - offset, idx + offset]:
                if 0 <= neighbor < self.n_replicas and neighbor != idx:
                    neighbor_E = self._get_replica_energy(self.replicas[neighbor])
                    if np.isfinite(neighbor_E):
                        self.replicas[idx] = self.replicas[neighbor].copy(copy_calc=False)
                        self.replicas[idx].set_calc(self.replicas[neighbor].calc)
                        self.logger.info(
                            "Replica %d reset from replica %d (E=%.3f)",
                            idx, neighbor, neighbor_E,
                        )
                        return
        # Fallback: reset from initial surface
        if self._initial_surface is not None:
            self.replicas[idx] = self._initial_surface.copy(copy_calc=False)
            self.replicas[idx].set_calc(self._initial_surface.calc)
            self.logger.info("Replica %d reset from initial surface", idx)

    def _reset_replica_from_checkpoint(self, idx: int) -> None:
        """Reset a replica from its last known good checkpoint."""
        if idx in self._checkpoints:
            self.replicas[idx] = self._checkpoints[idx].copy(copy_calc=False)
            self.replicas[idx].set_calc(self._checkpoints[idx].calc)
            self.logger.info("Replica %d reset from checkpoint", idx)
        else:
            self._reset_replica_from_neighbor(idx)

    def _update_checkpoints(self) -> None:
        """Save current state of healthy replicas as checkpoints."""
        for i in range(self.n_replicas):
            E = self._get_replica_energy(self.replicas[i])
            if np.isfinite(E):
                cp = self.replicas[i].copy(copy_calc=False)
                cp.set_calc(self.replicas[i].calc)
                self._checkpoints[i] = cp

    def _save_crash_dump(self, run_folder: Path, action: dict) -> None:
        """Save diagnostic info on abort for post-mortem analysis."""
        dump = {
            "action": action,
            "temperatures": self.temperatures,
            "replica_energies": [
                self._get_replica_energy(r) for r in self.replicas
            ],
        }
        dump_path = run_folder / "crash_dump.pkl"
        with open(dump_path, "wb") as f:
            pickle.dump(dump, f)
        self.logger.error("Crash dump saved to %s", dump_path)

    def run(
        self,
        surface: SurfaceSystem,
        total_sweeps: int = 100,
        sweep_size: int = 20,
        run_folder: str | Path | None = None,
        logger: logging.Logger | None = None,
        **kwargs,
    ) -> dict:
        """Run parallel tempering MC simulation with background monitoring.

        Args:
            surface: Initial SurfaceSystem (will be copied for each replica).
            total_sweeps: Number of MC sweeps per replica.
            sweep_size: Number of MC steps per sweep.
            run_folder: Directory for output files.
            logger: Logger instance.

        Returns:
            dict with per-replica histories and swap statistics.
        """
        if logger:
            self.logger = logger

        run_folder = Path(run_folder) if run_folder else Path("pt_run")
        run_folder.mkdir(parents=True, exist_ok=True)

        # Create replica subdirectories
        replica_folders = []
        for i in range(self.n_replicas):
            rf = run_folder / f"replica_{i}_T{self.temperatures[i]:.4f}"
            rf.mkdir(parents=True, exist_ok=True)
            replica_folders.append(rf)

        self._initial_surface = surface.copy(copy_calc=False)
        self._initial_surface.set_calc(surface.calc)
        self.replicas = self._create_replicas(surface)

        self.logger.info(
            "Starting Parallel Tempering: %d replicas, %d sweeps, %d steps/sweep",
            self.n_replicas, total_sweeps, sweep_size,
        )
        self.logger.info("Temperatures: %s", [f"{t:.4f}" for t in self.temperatures])

        # Start background monitor
        monitor = PTMonitor(config=self.monitor_config, logger=self.logger)
        monitor.start(self.n_replicas, self.temperatures)

        # Results tracking
        results = {
            "replica_energies": [[] for _ in range(self.n_replicas)],
            "replica_accept_rates": [[] for _ in range(self.n_replicas)],
            "replica_ads_counts": [[] for _ in range(self.n_replicas)],
            "replica_histories": [[] for _ in range(self.n_replicas)],
            "swap_accept_rates": [],
            "temperatures": self.temperatures,
            "monitor_events": [],
        }

        total_swaps_attempted = 0
        total_swaps_accepted = 0
        aborted = False

        try:
            for sweep_num in range(total_sweeps):
                self.logger.info("Sweep %d / %d", sweep_num + 1, total_sweeps)

                sweep_energies = []
                sweep_accept_rates = []

                # Run independent MC sweeps on all replicas
                for i in range(self.n_replicas):
                    sweep_result = self._sweep(
                        self.replicas[i], self.temperatures[i], sweep_size
                    )
                    results["replica_energies"][i].append(sweep_result["energy"])
                    results["replica_accept_rates"][i].append(sweep_result["acceptance_rate"])
                    results["replica_ads_counts"][i].append(sweep_result["adsorption_count"])

                    sweep_energies.append(sweep_result["energy"])
                    sweep_accept_rates.append(sweep_result["acceptance_rate"])

                    # Save structure snapshot
                    snapshot = self.replicas[i].copy(copy_calc=False)
                    snapshot.unset_calc()
                    results["replica_histories"][i].append(snapshot)

                # Attempt replica swaps
                swap_rate = None
                if (sweep_num + 1) % self.swap_interval == 0:
                    n_pairs = max(1, self.n_replicas // 2)
                    n_accepted = self._swap_round(sweep_num)
                    total_swaps_attempted += n_pairs
                    total_swaps_accepted += n_accepted

                    swap_rate = n_accepted / n_pairs
                    results["swap_accept_rates"].append(swap_rate)
                    self.logger.info(
                        "Swap round: %d/%d accepted (cumulative: %d/%d = %.1f%%)",
                        n_accepted, n_pairs,
                        total_swaps_accepted, total_swaps_attempted,
                        100 * total_swaps_accepted / max(1, total_swaps_attempted),
                    )

                # Post diagnostics to monitor
                monitor.post_sweep(sweep_num + 1, sweep_energies, sweep_accept_rates, swap_rate)

                # Check for monitor actions
                actions = monitor.get_actions()
                if actions:
                    results["monitor_events"].extend(actions)
                    if self._handle_actions(actions, run_folder):
                        aborted = True
                        break

                # Update checkpoints every 10 sweeps
                if (sweep_num + 1) % 10 == 0:
                    self._update_checkpoints()

                # Save structures periodically
                if (sweep_num + 1) % 10 == 0 or sweep_num == total_sweeps - 1:
                    for i in range(self.n_replicas):
                        self.replicas[i].save_structures(
                            sweep_num=sweep_num + 1,
                            save_folder=replica_folders[i],
                        )
        finally:
            monitor.stop()

        if aborted:
            self.logger.error("Simulation aborted by monitor — partial results saved")
        else:
            self.logger.info("Simulation completed normally")

        # Save final results
        self._save_results(results, run_folder)

        return results

    def _save_results(self, results: dict, run_folder: Path) -> None:
        """Save simulation results to disk."""
        import pandas as pd

        # Save per-replica statistics
        for i in range(self.n_replicas):
            df = pd.DataFrame({
                "energy": results["replica_energies"][i],
                "accept_rate": results["replica_accept_rates"][i],
                "ads_count": results["replica_ads_counts"][i],
            })
            df.to_csv(
                run_folder / f"replica_{i}_stats.csv",
                index=False, float_format="%.4f",
            )

        # Save swap statistics
        if results["swap_accept_rates"]:
            np.savetxt(
                run_folder / "swap_accept_rates.txt",
                results["swap_accept_rates"],
                fmt="%.4f",
            )

        # Save structures from the lowest-temperature replica (equilibrium samples)
        lowest_T_structures = results["replica_histories"][0]
        with open(run_folder / "lowest_T_structures.pkl", "wb") as f:
            pickle.dump(lowest_T_structures, f)

        self.logger.info("Results saved to %s", run_folder)

    def get_equilibrium_structures(self, results: dict, burn_in: int = 0) -> list:
        """Extract equilibrium structures from the lowest-temperature replica.

        Args:
            results: Output from run().
            burn_in: Number of initial sweeps to discard.

        Returns:
            List of SurfaceSystem snapshots after burn-in.
        """
        return results["replica_histories"][0][burn_in:]
