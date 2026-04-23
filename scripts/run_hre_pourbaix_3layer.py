"""HRE-MC driver with experimental-first ΔG_pbx + Multiple-Try Metropolis.

Differences from the original ``run_hre_pourbaix.py``:

  * ΔG₁ uses the 3-layer ``slab_correction`` (Layer A per-oxide Δ_O +
    Layer B per-species ZPE-TS + Layer C MDAnalysis H-bond count),
    enabled via ``use_adsorbate_gibbs=True`` (default on this branch).
    Replaces the legacy ``adsorbate_corrections={"HO": 0.23}`` path.

  * ΔG₂ optionally patches the loaded Pourbaix-diagram JSON via
    ``fork_port.apply_experimental_solid_overrides`` (e.g. shifting
    IrO₂(s) to Bratsch / Cordfunke −2.017 eV/fu). Requires the
    xiaochendu pymatgen fork, which is installed on Expanse. Skipped
    gracefully if ``fork_port`` can't import.

  * Sampling uses Multiple-Try Metropolis with k = ``--n_trials`` (default 4)
    candidates per step, Boltzmann-weighted pick, Liu 2000 Eq 2.4
    acceptance ratio (see ``HamiltonianREMC._mc_step_mtm``).

  * Structure snapshots every ``--save_interval`` sweeps (default 5,
    was hardcoded 10). 20 snapshots/replica × N replicas → dense enough
    for k-means clustering and Pourbaix-phase statistics.

Example:

    python scripts/run_hre_pourbaix_3layer.py \\
        --pH 0.0 \\
        --phi 0.0 0.5 0.9 1.2 1.4 1.5 1.6 1.8 2.0 \\
        --n_trials 4 --save_interval 5 \\
        --delta_O -0.1252 \\
        --metal Ir \\
        --experimental_override "IrO2:+1.76" \\
        --model /path/to/MACE-iro2-all.model \\
        --slab_pkl /path/to/pristine_clean.pkl \\
        --sites_pkl /path/to/sites_clean.pkl \\
        --phase_diagram /path/to/IrO_pd_fixed.json \\
        --pourbaix_diagram /path/to/IrO_pbx_mp.json \\
        --run_folder hre_mc_r1_3layer
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from time import perf_counter

import torch


def build_conditions(pH, phi, mc_temp, ec_temp):
    from mcmc.hamiltonian_re import ReplicaCondition
    return [
        ReplicaCondition(
            temperature=mc_temp, pH=ph, phi=p, electrochemical_temp=ec_temp
        )
        for ph in pH
        for p in phi
    ]


def patch_pbx_with_experimental_overrides(pbx_path: Path, overrides: dict, out_dir: Path) -> str:
    """Dump a new Pourbaix-diagram JSON with solid entries shifted to
    experimental ΔG°_f.

    HARD-REQUIRED for the experimental-first authoritative Gpbx pipeline:
    if ``fork_port`` cannot be imported or an override target is not found
    in the diagram, we ``sys.exit(1)`` rather than fall back to MP. Silent
    fallback to MP DFT+MP2020 solids is the exact failure mode the project
    is designed to avoid.
    """
    # fork_port lives at /home/james0001/project/pourbaix-diagram/fork_port.py
    # on local, or at containers/ on Expanse.
    for candidate in (
        Path("/home/james0001/project/pourbaix-diagram"),
        Path("/expanse/projects/qstore/csd807/gliu3/containers"),
    ):
        if (candidate / "fork_port.py").exists():
            sys.path.insert(0, str(candidate))
            break
    try:
        import fork_port  # noqa: F401 — auto-patches MontyDecoder
        from fork_port import apply_experimental_solid_overrides
        from monty.serialization import dumpfn, loadfn
    except ImportError as exc:
        sys.stderr.write(
            "\nFATAL: fork_port could not be imported. The experimental ΔG_pbx\n"
            f"pipeline requires fork_port + the xiaochendu pymatgen fork.\n"
            f"Underlying error: {exc}\n\n"
            "This script intentionally does NOT fall back to the legacy MP+MP2020\n"
            "solid-entry path. Fix the environment (install fork pymatgen + ensure\n"
            "fork_port.py is on a known search path) and re-run.\n"
        )
        sys.exit(1)

    pbx = loadfn(str(pbx_path))
    new_pbx, applied = apply_experimental_solid_overrides(pbx, overrides)
    missing = set(overrides) - set(applied)
    if missing:
        sys.stderr.write(
            f"\nFATAL: experimental override(s) {missing} not found as solid entries\n"
            f"in {pbx_path}. Either the formulas don't match the entries in the\n"
            "diagram, or the override is targeting an ion (apply_experimental_solid_overrides\n"
            "only shifts solids; MP ions are already on experimental NIST values).\n"
            "Aborting rather than silently dropping the override.\n"
        )
        sys.exit(1)
    logging.info("Applied experimental ΔG°_f overrides: %s", applied)
    out_path = out_dir / "pbx_with_exp_override.json"
    dumpfn(new_pbx, str(out_path))
    return str(out_path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Replica grid
    ap.add_argument("--pH", type=float, nargs="+", required=True)
    ap.add_argument("--phi", type=float, nargs="+", required=True)
    ap.add_argument("--mc_temp", type=float, default=0.5)
    ap.add_argument("--ec_temp", type=float, default=0.0257)

    # Sampling
    ap.add_argument("--total_sweeps", type=int, default=100)
    ap.add_argument("--sweep_size", type=int, default=20)
    ap.add_argument("--save_interval", type=int, default=5)
    ap.add_argument("--swap_interval", type=int, default=5)
    ap.add_argument("--n_trials", type=int, default=4,
                    help="MTM candidate count (1 = single-try Metropolis)")
    ap.add_argument("--max_snapshots_per_replica", type=int, default=None,
                    help="AL-aware early stop: break the sweep loop as soon "
                         "as every replica has saved this many snapshots. "
                         "Example: 5 → 9 replicas × 5 snapshots = 45 candidates "
                         "per AL round, regardless of total_sweeps. Default "
                         "None = run all total_sweeps.")

    # GMM-based uncertainty (single-MACE-model OOD detection) for smart AL.
    ap.add_argument("--gmm_training_xyz", default=None,
                    help="Path to an ASE-readable file (extxyz / traj / dir of "
                         "CIFs) containing the DFT training structures. If set, "
                         "a GaussianMixture is fit on their MACE embeddings at "
                         "startup, and MC is stopped early when newly-saved "
                         "snapshots' log-likelihood distribution has saturated "
                         "(no new OOD structures being found). Requires sklearn.")
    ap.add_argument("--gmm_n_components", type=int, default=16)
    ap.add_argument("--gmm_logl_threshold", type=float, default=None,
                    help="OOD cutoff on GMM log-L (below = OOD). If omitted, "
                         "auto-set to the 5%%-quantile of the training log-L.")
    ap.add_argument("--gmm_saturation_window", type=int, default=3)
    ap.add_argument("--gmm_saturation_tolerance", type=float, default=0.1)
    ap.add_argument("--gmm_min_saves_before_check", type=int, default=6)

    # 3-layer correction
    ap.add_argument("--delta_O", type=float, default=-0.1252,
                    help="Layer A per-O oxide correction in eV (default IrO2-fit)")
    ap.add_argument("--metal", default="Ir", help="Metal symbol for adsorbate geometry")
    ap.add_argument("--eps_hbond", type=float, default=-0.15,
                    help="Layer C H-bond correction in eV/H-bond")

    # Pourbaix-atoms source: controls how the ΔG₂ per-element partials are
    # constructed for each (pH, φ) point during MC.
    #   bratsch (default)  → pure experimental (Bratsch 1989 + Cordfunke 1981)
    #                        No IrO₄²⁻(aq) because Bratsch has no verified
    #                        value; IrO₃(s) covers high-φ. No MP / MP2020.
    #   mp                  → MP pourbaix_atoms via generate_pourbaix_atoms
    #                        (requires --experimental_override to shift MP
    #                        solids to experimental values). Legacy path;
    #                        use only to reproduce old runs.
    ap.add_argument("--pourbaix_source", choices=["bratsch", "mp"], default="bratsch",
                    help="Source of ΔG₂ standard-state energies. 'bratsch' "
                         "(default) uses the pure-experimental species list "
                         "from bulk_pourbaix_exp.ir_species_experimental. "
                         "'mp' uses MP pbx JSON + --experimental_override.")
    # Only required when --pourbaix_source=mp (legacy); optional on bratsch.
    ap.add_argument("--experimental_override", nargs="+", default=None,
                    help='Per-fu shifts applied to Pourbaix JSON solids, '
                         'e.g. --experimental_override IrO2:+1.76. '
                         "REQUIRED if --pourbaix_source mp; ignored on bratsch "
                         "(Bratsch uses experimental ΔG°_f directly without "
                         "needing MP JSON shifting).")

    # Files
    ap.add_argument("--model", required=True, help="MACE model .model path")
    ap.add_argument("--slab_pkl", required=True, help="Pickled ase.Atoms slab")
    ap.add_argument("--sites_pkl", required=True, help="Pickled dict with 'ads_coords'")
    ap.add_argument("--phase_diagram", required=True)
    ap.add_argument("--pourbaix_diagram", required=True)
    ap.add_argument("--elements", default="Ir,O")
    ap.add_argument("--adsorbates", default="O,HO")
    ap.add_argument("--surface_name", default="surface")

    # Runtime
    ap.add_argument("--run_folder", required=True)
    ap.add_argument("--cueq", action="store_true",
                    help="Enable cuEquivariance (off by default; some fine-tunes segfault)")
    args = ap.parse_args()

    # Prepare paths
    run_folder = Path(args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)

    # Surface-sampling repo on path (runner is in scripts/; mcmc is a sibling)
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from mcmc.calculators import MACEPourbaix
    from mcmc.hamiltonian_re import HamiltonianREMC, UncertaintyTracker
    from mcmc.system import SurfaceSystem
    from mcmc.utils import setup_logger

    logger = setup_logger("hre_3layer", run_folder / "hre.log", level=logging.INFO)
    logger.info("=" * 72)
    logger.info("feat/batched-ntrial — experimental-first ΔG_pbx + MTM")
    logger.info("=" * 72)
    logger.info("Replica grid: pH=%s φ=%s", args.pH, args.phi)
    logger.info("Sampling: %d sweeps × %d moves, save every %d, swap every %d, MTM k=%d",
                args.total_sweeps, args.sweep_size, args.save_interval,
                args.swap_interval, args.n_trials)
    logger.info("")
    logger.info("ΔG₁ side (MACEPourbaix, use_adsorbate_gibbs=True):")
    logger.info("  Layer A (bulk oxide): Δ_O = %+.4f eV/O applied to ALL O atoms", args.delta_O)
    from mcmc.corrections.adsorbate_gibbs import ADS_G_INTRINSIC
    logger.info("  Layer B (per-species adsorbate ZPE + ∫Cp dT − TS @ 298 K, ALL species):")
    for sp in ("O", "OH", "OOH", "H2O", "H"):
        if sp in ADS_G_INTRINSIC:
            logger.info("      *%-4s  G_intrinsic = %+.4f eV", sp, ADS_G_INTRINSIC[sp])
    logger.info("  Layer C (H-bonds): ε_HB = %+.3f eV × n_hbonds (Luzar-Chandler geometry)",
                args.eps_hbond)
    logger.info("")
    if args.pourbaix_source == "bratsch":
        logger.info("ΔG₂ side: pure Bratsch 1989 + Cordfunke 1981 experimental data")
        logger.info("  → No MP JSON; no MP2020; no Pourbaix-Atlas IrO₄²⁻ estimate.")
        logger.info("  → Ir species: {Ir(s), IrO₂(s), Ir³⁺(aq), IrO₃(s)} — IrO₃ at high φ")
    else:
        if not args.experimental_override:
            sys.stderr.write(
                "\nFATAL: --pourbaix_source mp requires --experimental_override "
                "(e.g. --experimental_override IrO2:+1.76). Otherwise ΔG₂ uses "
                "legacy MP DFT+MP2020 for solids. Use --pourbaix_source bratsch "
                "(default) to bypass MP entirely.\n"
            )
            sys.exit(1)
        logger.info("ΔG₂ side: LEGACY MP JSON + apply_experimental_solid_overrides(%s)",
                    ", ".join(args.experimental_override))
        logger.info("  → MP solid entries are shifted to experimental ΔG°_f.")
        logger.info("  → MP ion entries include Pourbaix-Atlas-estimated IrO₄²⁻.")
    logger.info("")
    logger.info("Legacy paths INACTIVE: adsorbate_corrections={HO:0.23} dict is skipped")
    logger.info("by use_adsorbate_gibbs=True in calculators.py; MP2020's -0.687 eV/O")
    logger.info("oxide_correction_per_O is skipped by the same flag.")
    logger.info("=" * 72)

    # Build the Pourbaix-diagram path for HamiltonianREMC. Only used by the
    # MP (legacy) path; bratsch path will override pourbaix_atoms on each
    # replica's MACEPourbaix via calc.set(pourbaix_atoms=...) and will not
    # rely on the diagram for ΔG₂.
    if args.pourbaix_source == "mp":
        overrides = {}
        for s in args.experimental_override:
            k, v = s.split(":")
            overrides[k] = float(v)
        pbx_path = patch_pbx_with_experimental_overrides(
            Path(args.pourbaix_diagram), overrides, run_folder,
        )
        logger.info("Using patched Pourbaix JSON (experimental ΔG°_f): %s", pbx_path)
    else:
        pbx_path = args.pourbaix_diagram  # loaded but overridden post-init

    # Load slab + sites
    with open(args.slab_pkl, "rb") as f:
        atoms = pickle.load(f)
    with open(args.sites_pkl, "rb") as f:
        ads_coords = pickle.load(f)
        if isinstance(ads_coords, dict) and "ads_coords" in ads_coords:
            ads_coords = ads_coords["ads_coords"]
    logger.info("Slab: %s (%d atoms), %d ads sites",
                atoms.get_chemical_formula(), len(atoms), len(ads_coords))

    # Calculator (3-layer authoritative path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = MACEPourbaix(args.model, device=device, enable_cueq=args.cueq)
    calc.set(
        temperature=args.ec_temp,
        pH=args.pH[0],
        phi=args.phi[0],
        use_adsorbate_gibbs=True,
        slab_correction_kwargs={
            "delta_O": args.delta_O,
            "metal_symbol": args.metal,
            "eps_hbond": args.eps_hbond,
        },
    )
    logger.info("MACEPourbaix: use_adsorbate_gibbs=%s", calc.use_adsorbate_gibbs)

    surface = SurfaceSystem(
        atoms,
        calc=calc,
        ads_coords=ads_coords,
        occ=[0] * len(ads_coords),
        system_settings={"surface_name": args.surface_name, "cutoff": 6.0,
                         "surface_depth": None},
        save_folder=str(run_folder),
    )

    uncertainty = UncertaintyTracker(threshold=0.15, batch_size=10, logger=logger)

    hre = HamiltonianREMC(
        conditions=build_conditions(args.pH, args.phi, args.mc_temp, args.ec_temp),
        adsorbates=args.adsorbates.split(","),
        phase_diagram_path=args.phase_diagram,
        pourbaix_diagram_path=pbx_path,
        elements=args.elements.split(","),
        canonical=False,
        swap_interval=args.swap_interval,
        n_trials=args.n_trials,
        uncertainty_tracker=uncertainty,
        pourbaix_source=args.pourbaix_source,
        logger=logger,
    )

    t0 = perf_counter()
    # Build GMM tracker if the user supplied a training set. Fit happens
    # once here before the MC run; each save during the run feeds snapshots
    # in via add_snapshot() and may trigger early stop on saturation.
    gmm_tracker = None
    if args.gmm_training_xyz:
        from ase.io import read as ase_read
        from mcmc.uncertainty.gmm_tracker import GMMUncertaintyTracker
        training_atoms = ase_read(args.gmm_training_xyz, index=":")
        if not isinstance(training_atoms, list):
            training_atoms = [training_atoms]
        logger.info("GMM tracker: loaded %d training structures from %s",
                    len(training_atoms), args.gmm_training_xyz)
        gmm_tracker = GMMUncertaintyTracker(
            calc=calc,
            training_atoms=training_atoms,
            n_components=args.gmm_n_components,
            logl_threshold=args.gmm_logl_threshold,
            saturation_window=args.gmm_saturation_window,
            saturation_tolerance=args.gmm_saturation_tolerance,
            min_saves_before_check=args.gmm_min_saves_before_check,
            logger=logger,
        )

    results = hre.run(
        surface=surface,
        total_sweeps=args.total_sweeps,
        sweep_size=args.sweep_size,
        save_interval=args.save_interval,
        max_snapshots_per_replica=args.max_snapshots_per_replica,
        gmm_tracker=gmm_tracker,
        run_folder=str(run_folder),
        logger=logger,
    )
    elapsed = perf_counter() - t0

    logger.info("HRE-MC done in %.1f s (%.2f h)", elapsed, elapsed / 3600.0)

    diagram = hre.get_pourbaix_diagram_data(results)
    logger.info("Diagram: %d grid points", len(diagram))
    for (pH, phi), data in sorted(diagram.items()):
        comp = data.get("composition", {})
        comp_str = " ".join(f"{k}{v}" for k, v in sorted(comp.items()))
        logger.info(
            "(pH=%.1f φ=%.2f) E=%+.3f eV #ads=%d %s",
            pH, phi, data["energy"], data["ads_count"], comp_str,
        )

    # Write a summary marker so /loop monitoring can see completion
    done_text = (
        f"elapsed_s={elapsed:.1f}\nn_replicas={len(hre.conditions)}\n"
        f"n_diagram_points={len(diagram)}\n"
    )
    if gmm_tracker is not None:
        s = gmm_tracker.summary()
        done_text += (
            f"gmm_n_snapshots={s['n_snapshots']}\n"
            f"gmm_n_ood={s['n_ood']}\n"
            f"gmm_logl_threshold={s['logl_threshold']:.4f}\n"
            f"gmm_logl_min={s.get('logl_min', float('nan')):.4f}\n"
            f"gmm_ood_queue_size={s['ood_queue_size']}\n"
        )
        # Save OOD candidate list (CIFs) for downstream AL relabeling
        from ase.io import write as ase_write
        ood_dir = run_folder / "ood_candidates"
        ood_dir.mkdir(exist_ok=True)
        for i, atoms in enumerate(gmm_tracker.ood_queue):
            ase_write(ood_dir / f"ood_{i:03d}.cif", atoms)
        logger.info("GMM tracker: wrote %d OOD candidates to %s",
                    len(gmm_tracker.ood_queue), ood_dir)
    (run_folder / "DONE").write_text(done_text)


if __name__ == "__main__":
    main()
