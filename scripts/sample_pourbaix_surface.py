"""Surface sampling under aqueous electrochemical conditions using VSSR-MC."""

import argparse
import json
import pickle
from logging import getLevelNamesMapping
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
import torch
from default_settings import DEFAULT_CUTOFFS, DEFAULT_SAMPLING_SETTINGS
from monty.serialization import dumpfn, loadfn
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure

from mcmc import MCMC
from mcmc.calculators import MACEPourbaix
from mcmc.pourbaix.atoms import generate_pourbaix_atoms
from mcmc.system import SurfaceSystem
from mcmc.utils import setup_logger
from mcmc.utils.plot import plot_summary_stats
from mcmc.utils.setup import setup_folders

np.set_printoptions(precision=3, suppress=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform MCMC on surfaces under electrochemical conditions."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="LaMnO3_001_2x2",
        help="Name of the run",
    )
    parser.add_argument(
        "--starting_structure_path",
        type=str,
        help="path to the starting structure",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="path to the MACE model file",
    )
    parser.add_argument(
        "--phase_diagram_path",
        type=str,
        help="path to the saved pymatgen PhaseDiagram",
    )
    parser.add_argument(
        "--pourbaix_diagram_path",
        type=str,
        help="path to the saved pymatgen PourbaixDiagram",
    )
    parser.add_argument(
        "--settings_path",
        type=str,
        default="settings.json",
        help="Path to the settings file",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to use for calculations",
    )
    parser.add_argument(
        "--pH",
        type=float,
        default=None,
        help="Override pH in calc_settings (sweep pH from command line)",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=None,
        help="Override electric potential (phi) in calc_settings (sweep phi from command line)",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level",
    )
    return parser.parse_args()


def main(
    run_name: str,
    starting_structure_path: Path | str,
    model_path: str,
    phase_diagram_path: Path | str,
    pourbaix_diagram_path: Path | str,
    settings_path: str = "settings.json",
    device: Literal["cpu", "cuda"] = "cuda",
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = "info",
    pH: float | None = None,
    phi: float | None = None,
) -> None:
    """Perform VSSR-MC sampling for surfaces.

    Args:
        run_name (str): name of the run
        starting_structure_path (Union[Path, str]): path to the starting structure
        model_path (str): path to the MACE model file
        phase_diagram_path (Union[Path, str]): path to the saved pymatgen PhaseDiagram
        pourbaix_diagram_path (Union[Path, str]): path to the saved pymatgen PourbaixDiagram
        settings_path (str, optional): path to the settings file. Defaults to "settings.json"
        device (Literal["cpu", "cuda"], optional): device to use for calculations.
            Defaults to "cuda"
        logging_level (Literal["debug", "info", "warning", "error", "critical"], optional): logging
        pH (float, optional): override pH in calc_settings. Defaults to None (use config value).
        phi (float, optional): override phi in calc_settings. Defaults to None (use config value).
    """
    # Load settings
    all_settings = loadfn(settings_path)
    calc_settings, system_settings, sampling_settings = (
        all_settings["calc_settings"],
        all_settings["system_settings"],
        all_settings["sampling_settings"],
    )

    # Apply CLI overrides for pH and phi
    if pH is not None:
        calc_settings["pH"] = pH
    if phi is not None:
        calc_settings["phi"] = phi

    # Update empty settings with default params
    system_settings["surface_name"] = system_settings.get("surface_name", run_name)
    system_settings["cutoff"] = system_settings.get("cutoff", DEFAULT_CUTOFFS["MACE"])
    sampling_settings = DEFAULT_SAMPLING_SETTINGS | sampling_settings

    # Initialize run folder
    if not sampling_settings.get("run_folder"):
        run_folder = setup_folders(
            system_settings["surface_name"],
            canonical=sampling_settings["canonical"],
            total_sweeps=sampling_settings["total_sweeps"],
            start_temp=sampling_settings["start_temp"],
            alpha=sampling_settings["alpha"],
            pH=calc_settings["pH"],
            phi=calc_settings["phi"],
        )
        sampling_settings["run_folder"] = run_folder
    else:
        run_folder = Path(sampling_settings["run_folder"])
        run_folder.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = setup_logger(
        "mcmc", run_folder / "mc.log", level=getLevelNamesMapping()[logging_level.upper()]
    )

    # Load offset data if offset path is provided
    if "offset_data" in calc_settings:
        offset_data = calc_settings["offset_data"]
        if isinstance(offset_data, str | Path):
            try:
                with open(offset_data, "r", encoding="utf-8") as f:
                    offset_data = json.load(f)
                    calc_settings["offset_data"] = offset_data
            except FileNotFoundError as e:
                logger.error("Offset data file not found at the provided path.")
                raise e

    # Load prepared pristine slab
    try:
        with open(starting_structure_path, "rb") as f:
            starting_slab = pickle.load(f)
    except FileNotFoundError as e:
        logger.error("Pristine surface pkl file not found.")
        raise e

    # Obtain Pourbaix atoms
    chem_symbols = starting_slab.get_chemical_symbols()
    elements = list(set(chem_symbols))
    logger.info("Elements: %s", elements)

    if "pourbaix_atoms" not in calc_settings:
        pourbaix_atoms = generate_pourbaix_atoms(
            phase_diagram_path,
            pourbaix_diagram_path,
            calc_settings["phi"],
            calc_settings["pH"],
            elements,
        )
        calc_settings["pourbaix_atoms"] = pourbaix_atoms
        logger.info("Generated Pourbaix atoms: %s", pourbaix_atoms)
    else:
        pourbaix_atoms = calc_settings["pourbaix_atoms"]
        logger.info("Using provided Pourbaix atoms: %s", pourbaix_atoms)

    # Save updated run settings
    all_settings = {
        "system_settings": system_settings,
        "sampling_settings": sampling_settings,
        "calc_settings": calc_settings,
    }
    dumpfn(all_settings, run_folder / "settings.json", indent=4)

    # Obtain adsorption sites
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
    logger.info("Generated adsorption coordinates are: %s...", ads_positions[:5])

    if system_settings.get("sample_surface_atoms", False):
        logger.info("Sampling surface atoms")
        surf_atom_idx = starting_slab.get_surface_atoms()
        surf_atom_positions = starting_slab.get_positions()[surf_atom_idx]
        logger.info("Surface atom coordinates are: %s...", surf_atom_positions[:5])

        # Get surface atom coordinates
        all_ads_coords = np.vstack([surf_atom_positions, ads_positions])

        # Set occupation array
        occ = np.hstack(
            [
                surf_atom_idx,
                [0] * len(ads_positions),
            ]
        )

        # Set corresponding adsorbate group array
        mask = np.isin(np.arange(len(starting_slab)), surf_atom_idx)
        ads_group = mask * np.arange(len(starting_slab))
    else:
        logger.info("Not sampling surface atoms")
        all_ads_coords = ads_positions
        occ = [0] * len(ads_positions)
        ads_group = [0] * len(starting_slab)

    logger.info("Starting occupation array: %s...", occ[:5])
    starting_slab.set_array("ads_group", ads_group, dtype=int)
    logger.info("Starting adsorbate group array: %s...", starting_slab.get_array("ads_group")[:5])

    # Initialize Calculator
    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"

    mace_pourbaix_calc = MACEPourbaix(
        model_path,
        device=device,
        enable_cueq=True,
    )
    mace_pourbaix_calc.set(**calc_settings)

    # Initialize SurfaceSystem
    surface = SurfaceSystem(
        starting_slab,
        calc=mace_pourbaix_calc,
        ads_coords=all_ads_coords,
        occ=occ,
        system_settings=system_settings,
        save_folder=run_folder,
    )
    surface.all_atoms.write(run_folder / "all_virtual_ads.cif")
    logger.info("Starting surface energy: %.3f eV", float(surface.get_surface_energy()))

    # Perform MCMC
    mcmc = MCMC(**sampling_settings)
    start = perf_counter()
    results = mcmc.run(
        surface=surface,
        **sampling_settings,
    )
    stop = perf_counter()
    logger.info("Time taken = %.3f seconds", stop - start)

    # Save SurfaceSystem objects for later use in latent space clustering or analysis
    structures = results["history"]
    with open(run_folder / f"{len(structures)}_mcmc_structures.pkl", "wb") as f:
        pickle.dump(structures, f)
    logger.info("Saving all %d surfaces", len(structures))

    # Save relaxation trajectories
    trajectories = results["trajectories"]
    traj_structures = [traj_info["atoms"] for traj_info in trajectories]
    traj_structures = [
        item for sublist in traj_structures for item in sublist
    ]  # flatten nested list
    with open(run_folder / f"{len(traj_structures)}_relaxation_structures.pkl", "wb") as f:
        pickle.dump(traj_structures, f)
    logger.info("Saving all %d slabs in relaxation trajectories", len(traj_structures))

    # Save statistics in csv
    stats_df = pd.DataFrame(
        {
            "energy": results["energy_hist"],
            "frac_accept": results["frac_accept_hist"],
            "adsorption_count": results["adsorption_count_hist"],
        }
    )
    stats_df.to_csv(run_folder / "stats.csv", index=False, float_format="%.3f")
    logger.info("Saving statistics in csv")

    # Plot statistics
    plot_summary_stats(
        results["energy_hist"],
        results["frac_accept_hist"],
        results["adsorption_count_hist"],
        mcmc.total_sweeps,
        save_folder=run_folder,
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.run_name,
        args.starting_structure_path,
        args.model_path,
        args.phase_diagram_path,
        args.pourbaix_diagram_path,
        args.settings_path,
        args.device,
        args.logging_level,
        args.pH,
        args.phi,
    )
