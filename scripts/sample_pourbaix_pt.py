"""Surface Pourbaix sampling with Parallel Tempering (Replica Exchange) MC.

Usage:
    python sample_pourbaix_pt.py \
        --run_name CuSn_001 \
        --starting_structure_path pristine.pkl \
        --model_path MACE-matpes-pbe-omat-ft.model \
        --phase_diagram_path CuSn_pd.json \
        --pourbaix_diagram_path CuSn_pbx.json \
        --settings_path configs/sample_cusn_pourbaix_config.json \
        --temperatures 0.01 0.05 0.1 0.3 0.5 \
        --swap_interval 1 \
        --pH 7.0 --phi 0.0
"""

import argparse
import json
import pickle
from logging import getLevelNamesMapping
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from default_settings import DEFAULT_CUTOFFS, DEFAULT_SAMPLING_SETTINGS
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure

from mcmc.calculators import MACEPourbaix
from mcmc.parallel_tempering import ParallelTemperingMC
from mcmc.pourbaix.atoms import generate_pourbaix_atoms
from mcmc.system import SurfaceSystem
from mcmc.utils import setup_logger
from mcmc.utils.setup import setup_folders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel Tempering MC for surface Pourbaix sampling."
    )
    parser.add_argument("--run_name", type=str, default="CuSn_001_PT")
    parser.add_argument("--starting_structure_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--phase_diagram_path", type=str, required=True)
    parser.add_argument("--pourbaix_diagram_path", type=str, required=True)
    parser.add_argument("--settings_path", type=str, default="settings.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--pH", type=float, default=None)
    parser.add_argument("--phi", type=float, default=None)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1, 0.3, 0.5],
        help="Replica temperatures in kT (eV). Default: 0.01 0.05 0.1 0.3 0.5",
    )
    parser.add_argument(
        "--swap_interval",
        type=int,
        default=1,
        help="Attempt replica swaps every N sweeps (default: 1)",
    )
    parser.add_argument(
        "--swap_scheme",
        choices=["sequential", "random"],
        default="sequential",
        help="Swap scheme: sequential (even/odd alternation) or random pair",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )
    return parser.parse_args()


def main(
    run_name: str,
    starting_structure_path: str,
    model_path: str,
    phase_diagram_path: str,
    pourbaix_diagram_path: str,
    settings_path: str = "settings.json",
    device: Literal["cpu", "cuda"] = "cuda",
    logging_level: str = "info",
    pH: float | None = None,
    phi: float | None = None,
    temperatures: list[float] | None = None,
    swap_interval: int = 1,
    swap_scheme: str = "sequential",
) -> dict:
    """Run Parallel Tempering MC for surface Pourbaix sampling.

    Args:
        run_name: Name for this run.
        starting_structure_path: Path to pristine slab pickle file.
        model_path: Path to MACE model.
        phase_diagram_path: Path to pymatgen PhaseDiagram JSON.
        pourbaix_diagram_path: Path to pymatgen PourbaixDiagram JSON.
        settings_path: Path to settings JSON.
        device: "cpu" or "cuda".
        logging_level: Logging verbosity.
        pH: Override pH from settings.
        phi: Override electric potential from settings.
        temperatures: List of replica temperatures in kT (eV).
        swap_interval: Attempt swaps every N sweeps.
        swap_scheme: "sequential" or "random".

    Returns:
        dict: Simulation results.
    """
    if temperatures is None:
        temperatures = [0.01, 0.05, 0.1, 0.3, 0.5]

    # Load settings
    all_settings = loadfn(settings_path)
    calc_settings = all_settings["calc_settings"]
    system_settings = all_settings["system_settings"]
    sampling_settings = all_settings["sampling_settings"]

    if pH is not None:
        calc_settings["pH"] = pH
    if phi is not None:
        calc_settings["phi"] = phi

    system_settings["surface_name"] = system_settings.get("surface_name", run_name)
    system_settings["cutoff"] = system_settings.get("cutoff", DEFAULT_CUTOFFS["MACE"])
    sampling_settings = DEFAULT_SAMPLING_SETTINGS | sampling_settings

    # Setup run folder
    T_str = "_".join(f"{t:.3f}" for t in sorted(temperatures))
    run_folder = Path(f"runs/{run_name}_PT_pH{calc_settings['pH']}_phi{calc_settings['phi']}")
    run_folder.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        "pt_mc", run_folder / "pt_mc.log",
        level=getLevelNamesMapping()[logging_level.upper()],
    )

    # Load offset data
    if "offset_data" in calc_settings:
        offset_data = calc_settings["offset_data"]
        if isinstance(offset_data, str | Path):
            with open(offset_data, "r", encoding="utf-8") as f:
                calc_settings["offset_data"] = json.load(f)

    # Load pristine slab
    with open(starting_structure_path, "rb") as f:
        starting_slab = pickle.load(f)

    # Generate Pourbaix atoms
    elements = list(set(starting_slab.get_chemical_symbols()))
    logger.info("Elements: %s", elements)

    if "pourbaix_atoms" not in calc_settings:
        pourbaix_atoms = generate_pourbaix_atoms(
            phase_diagram_path, pourbaix_diagram_path,
            calc_settings["phi"], calc_settings["pH"], elements,
        )
        calc_settings["pourbaix_atoms"] = pourbaix_atoms
        logger.info("Generated Pourbaix atoms: %s", pourbaix_atoms)

    # Save settings
    dumpfn(
        {"system_settings": system_settings, "sampling_settings": sampling_settings,
         "calc_settings": calc_settings, "temperatures": temperatures,
         "swap_interval": swap_interval, "swap_scheme": swap_scheme},
        run_folder / "settings.json", indent=4,
    )

    # Find adsorption sites
    starting_pmg_slab = Structure.from_ase_atoms(starting_slab)
    site_finder = AdsorbateSiteFinder(starting_pmg_slab)
    all_ads_positions = site_finder.find_adsorption_sites(
        put_inside=True,
        symm_reduce=system_settings.get("symm_reduce", False),
        near_reduce=system_settings.get("near_reduce", 0.01),
        distance=system_settings.get("planar_distance", 2.0),
        no_obtuse_hollow=system_settings.get("no_obtuse_hollow", True),
    )
    ads_positions = all_ads_positions[system_settings.get("ads_site_type", "all")]

    if system_settings.get("sample_surface_atoms", False):
        surf_atom_idx = starting_slab.get_surface_atoms()
        surf_atom_positions = starting_slab.get_positions()[surf_atom_idx]
        all_ads_coords = np.vstack([surf_atom_positions, ads_positions])
        occ = np.hstack([surf_atom_idx, [0] * len(ads_positions)])
        mask = np.isin(np.arange(len(starting_slab)), surf_atom_idx)
        ads_group = mask * np.arange(len(starting_slab))
    else:
        all_ads_coords = ads_positions
        occ = [0] * len(ads_positions)
        ads_group = [0] * len(starting_slab)

    starting_slab.set_array("ads_group", ads_group, dtype=int)

    # Initialize calculator
    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    mace_calc = MACEPourbaix(model_path, device=device, enable_cueq=True)
    mace_calc.set(**calc_settings)

    # Initialize surface
    surface = SurfaceSystem(
        starting_slab, calc=mace_calc, ads_coords=all_ads_coords,
        occ=occ, system_settings=system_settings, save_folder=run_folder,
    )
    surface.all_atoms.write(run_folder / "all_virtual_ads.cif")
    logger.info("Starting surface energy: %.3f eV", float(surface.get_surface_energy()))

    # Run Parallel Tempering
    pt_mc = ParallelTemperingMC(
        temperatures=temperatures,
        adsorbates=sampling_settings.get("adsorbates", []),
        canonical=sampling_settings.get("canonical", False),
        swap_interval=swap_interval,
        swap_scheme=swap_scheme,
        logger=logger,
    )

    start = perf_counter()
    results = pt_mc.run(
        surface=surface,
        total_sweeps=sampling_settings.get("total_sweeps", 100),
        sweep_size=sampling_settings.get("sweep_size", 20),
        run_folder=run_folder,
        logger=logger,
    )
    elapsed = perf_counter() - start
    logger.info("Parallel Tempering completed in %.1f seconds", elapsed)

    # Print summary
    logger.info("=== Summary ===")
    for i, T in enumerate(temperatures):
        energies = results["replica_energies"][i]
        rates = results["replica_accept_rates"][i]
        logger.info(
            "Replica %d (T=%.4f): final E=%.3f, mean accept=%.1f%%",
            i, T, energies[-1], 100 * np.mean(rates),
        )
    if results["swap_accept_rates"]:
        logger.info(
            "Swap acceptance: mean=%.1f%%",
            100 * np.mean(results["swap_accept_rates"]),
        )

    return results


if __name__ == "__main__":
    args = parse_args()
    main(
        run_name=args.run_name,
        starting_structure_path=args.starting_structure_path,
        model_path=args.model_path,
        phase_diagram_path=args.phase_diagram_path,
        pourbaix_diagram_path=args.pourbaix_diagram_path,
        settings_path=args.settings_path,
        device=args.device,
        logging_level=args.logging_level,
        pH=args.pH,
        phi=args.phi,
        temperatures=args.temperatures,
        swap_interval=args.swap_interval,
        swap_scheme=args.swap_scheme,
    )
