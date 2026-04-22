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

    # 3-layer correction
    ap.add_argument("--delta_O", type=float, default=-0.1252,
                    help="Layer A per-O oxide correction in eV (default IrO2-fit)")
    ap.add_argument("--metal", default="Ir", help="Metal symbol for adsorbate geometry")
    ap.add_argument("--eps_hbond", type=float, default=-0.15,
                    help="Layer C H-bond correction in eV/H-bond")

    # Experimental ΔG°_f overrides (ΔG₂ side) — REQUIRED for the authoritative
    # experimental-first pipeline. We do not allow MC to run without the
    # overrides: ΔG₂ on stock MP data uses DFT+MP2020 for solids (inconsistent
    # with the 3-layer per-oxide Δ_O on the ΔG₁ side and with the MP ion
    # entries themselves, which ARE experimental). Set to e.g.
    # "IrO2:+1.76" so the IrO₂(s) solid energy shifts to Bratsch /
    # Cordfunke −2.017 eV/fu.
    ap.add_argument("--experimental_override", nargs="+", required=True,
                    help='REQUIRED per-fu shifts applied to Pourbaix JSON solids, '
                         'e.g. --experimental_override IrO2:+1.76. '
                         "Requires fork_port (xiaochendu pymatgen fork). "
                         "No implicit fallback to MP DFT+MP2020 solids.")

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
    logger.info("feat/batched-ntrial — 3-layer ΔG_pbx + MTM (k=%d)", args.n_trials)
    logger.info("Replica grid: pH=%s φ=%s", args.pH, args.phi)
    logger.info("Sampling: %d sweeps × %d moves, save every %d sweeps, swap every %d",
                args.total_sweeps, args.sweep_size, args.save_interval, args.swap_interval)
    logger.info("3-layer: Δ_O=%.4f eV/O, metal=%s, ε_HB=%.3f eV",
                args.delta_O, args.metal, args.eps_hbond)

    # Experimental ΔG°_f override on the Pourbaix JSON (REQUIRED; argparse
    # enforces at least one value — see required=True on the arg).
    overrides = {}
    for s in args.experimental_override:
        k, v = s.split(":")
        overrides[k] = float(v)
    pbx_path = patch_pbx_with_experimental_overrides(
        Path(args.pourbaix_diagram), overrides, run_folder,
    )
    logger.info("Using patched Pourbaix JSON (experimental ΔG°_f): %s", pbx_path)

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
        logger=logger,
    )

    t0 = perf_counter()
    results = hre.run(
        surface=surface,
        total_sweeps=args.total_sweeps,
        sweep_size=args.sweep_size,
        save_interval=args.save_interval,
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
    (run_folder / "DONE").write_text(
        f"elapsed_s={elapsed:.1f}\nn_replicas={len(hre.conditions)}\n"
        f"n_diagram_points={len(diagram)}\n"
    )


if __name__ == "__main__":
    main()
