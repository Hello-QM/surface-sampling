"""Create pymatgen surface formation energy entries from VSSR-MC sampled surfaces"""

import argparse
import pickle as pkl
from datetime import datetime
from logging import getLevelNamesMapping
from pathlib import Path
from typing import Literal

import ase
import numpy as np
import torch
from monty.serialization import loadfn
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Structure
from pymatgen.entries.compatibility import (
    MaterialsProject2020Compatibility,
    MaterialsProjectAqueousCompatibility,
)
from pymatgen.entries.computed_entries import ComputedStructureEntry

from mcmc.calculators import MACESurface, get_results_single
from mcmc.dynamics import optimize_slab
from mcmc.pourbaix.utils import SurfaceOHCompatibility
from mcmc.utils import setup_logger
from mcmc.utils.misc import load_dataset_from_files

np.set_printoptions(precision=3, suppress=True)

HARTREE_TO_EV = 27.211386245988

SYMBOLS = {
    "La": "PAW_PBE La 06Sep2000",
    "O": "PAW_PBE O 08Apr2002",
    "Ir": "PAW_PBE Ir 06Sep2000",
    "Pt": "PAW_PBE Pt 04Feb2005",
    "Mn": "PAW_PBE Mn_pv 02Aug2007",
    "H": "PAW_PBE H 15Jun2001",
}

DFT_U_VALUES = {
    "La": 0.0,
    "Mn": 3.9,
    "Pt": 0.0,  # no need for metals
    "O": 0.0,
    "Ir": 0.0,
    "H": 0.0,
}

DEFAULT_OH_ZPE_TS_CORRECTION = 0.23  # eV, from Rong and Kolpak, J. Phys. Chem. Lett., 2015
DEFAULT_HYDROGEN_BOND_CORRECTION = -0.30  # eV, from Calle-Vallejo et al., 2011

O2_DFT_ENERGY = -4.94795546875  # DFT energy before any entropy correction
H2O_DFT_ENERGY = -5.192751548333333  # DFT energy before any entropy correction
H2O_ADJUSTMENTS = -0.229  # already counted in the H2O energy


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create pymatgen surface formation energy Entry's from VSSR-MC sampled surfaces "
            "under electrochemical conditions."
        )
    )
    parser.add_argument(
        "--surface_name",
        type=str,
        default="LaMnO3_001_2x2",
        help="name of the surface",
    )
    parser.add_argument(
        "--file_paths",
        nargs="+",
        help="Full paths to pickle or XYZ files of structures.",
        type=Path,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["MACE", "DFT"],
        default="MACE",
        help="type of model to use",
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="*",
        default=[""],
        help="paths to MACE model files",
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
        "--elements",
        nargs="+",
        type=str,
        help="list of elements",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="./",
        help="Folder to output.",
    )
    parser.add_argument("--relax", action="store_true", help="perform relaxation for the steps")
    parser.add_argument("--relax_steps", type=int, default=5, help="max relaxation steps")
    parser.add_argument(
        "--correct_hydroxide_energy",
        action="store_true",
        help="correct hydroxide energy (add ZPE-TS)",
    )
    parser.add_argument(
        "--correct_hydrogen_bond_energy",
        action="store_true",
        help="correct hydrogen bond energy",
    )
    parser.add_argument(
        "--aq_compat",
        action="store_true",
        help="use MaterialsProjectAqueousCompatibility",
    )
    parser.add_argument(
        "--input_slab_name",
        action="store_true",
        help="Input stoichiometry of the slab as the slab name",
    )
    parser.add_argument(
        "--input_job_id",
        action="store_true",
        help="Input job ID as the slab name",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to use for calculations",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level",
    )

    return parser.parse_args()


def get_params(elements: list[str]) -> dict:
    """Get the parameters for the ComputedStructureEntry.

    Args:
        elements (list[str]): list of elements

    Returns:
        dict: parameters for the ComputedStructureEntry
    """
    return {
        "run_type": "GGA+U",
        "is_hubbard": True,
        "hubbards": {elem: DFT_U_VALUES[elem] for elem in elements},
        "potcar_symbols": [SYMBOLS[elem] for elem in elements],
    }


def create_computed_entry(
    slab: ase.Atoms, energy: float, slab_name: str | None = None
) -> ComputedStructureEntry:
    """Create a ComputedStructureEntry from an ASE Atoms object.

    Args:
        slab (Atoms): surface slab
        energy (float): raw predicted energy of the slab in eV (preferably with MP 2020 corrections)
        slab_name (str): name of the slab

    Returns:
        ComputedStructureEntry: ComputedStructureEntry object
    """
    pmg_struct = Structure.from_ase_atoms(slab)
    params = get_params(list(set(slab.get_chemical_symbols())))
    return ComputedStructureEntry(pmg_struct, energy, parameters=params, entry_id=slab_name)


def create_surface_formation_entry(
    raw_energy_entry: ComputedStructureEntry, phase_diagram: PhaseDiagram
) -> ComputedStructureEntry:
    """Create a ComputedStructureEntry with surface formation energy. Required as input to
    SurfacePourbaixDiagram.

    Args:
        raw_energy_entry (ComputedStructureEntry): Entry with the raw energies in eV (preferably
            with MP 2020 corrections)
        phase_diagram (PhaseDiagram): pymatgen PhaseDiagram object for the relevant atomic species

    Returns:
        ComputedStructureEntry: ComputedStructureEntry object with surface formation energy
    """
    return ComputedStructureEntry(
        raw_energy_entry,
        phase_diagram.get_form_energy(raw_energy_entry),
        parameters=raw_energy_entry.parameters,
        entry_id=raw_energy_entry.entry_id,
    )


def main(
    surface_name: str,
    file_names: list[str],
    model_type: Literal["MACE", "DFT"],
    model_paths: list[str],
    phase_diagram_path: Path | str,
    pourbaix_diagram_path: Path | str,
    correct_hydroxide_energy: bool = False,
    correct_hydrogen_bond_energy: bool = False,
    aq_compat: bool = False,
    input_slab_name: bool = False,
    input_job_id: bool = False,
    device: str = "cuda",
    relax: bool = False,
    relax_steps: int = 20,
    save_folder: str = "./",
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = "info",
) -> None:
    """Create pymatgen ComputedEntries for surface formation energies. Uses MACE models to predict
    energies. Relaxes the structures if relax is True.

    Args:
        surface_name (str): name of the surface
        file_names (list[str]): list of file paths to the ASE Atoms objects
        model_type (Literal["MACE", "DFT"]): type of model to use
        model_paths (list[str]): list of paths to MACE model files
        phase_diagram_path (Path | str): path to the saved pymatgen PhaseDiagram
        pourbaix_diagram_path (Path | str): path to the saved pymatgen PourbaixDiagram
        correct_hydroxide_energy (bool, optional): correct hydroxide energy (add ZPE-TS). Defaults
            to False.
        correct_hydrogen_bond_energy (bool, optional): correct hydrogen bond energy.
            Defaults to False.
        aq_compat (bool, optional): use MaterialsProjectAqueousCompatibility. Defaults to False.
        input_slab_name (bool, optional): Input stoichiometry of the slab as the slab name. Defaults
            to False.
        input_job_id (bool, optional): Input job ID as the slab name. Defaults to False.
        device (str, optional): device to use for calculations. Defaults to "cuda".
        relax (bool, optional): perform relaxation for the steps. Defaults to False.
        relax_steps (int, optional): max relaxation steps. Defaults to 20.
        save_folder (str, optional): folder to output. Defaults to "./".
        logging_level (Literal["debug", "info", "warning", "error", "critical"], optional):
            logging level. Defaults to "info".
    """
    start_timestamp = datetime.now().isoformat(sep="-", timespec="milliseconds")

    # Initialize save folder
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    file_base = f"{start_timestamp}_generate_surface_formation_entries_{surface_name}"

    # Initialize logger
    logger = setup_logger(
        "generate_formation_entries",
        save_path / "generate_formation_entries.log",
        level=getLevelNamesMapping()[logging_level.upper()],
    )

    logger.info("There are a total of %d input files", len(file_names))
    all_structures = load_dataset_from_files(file_names)
    # If all_structures are SurfaceSystems, take the relaxed_atoms
    all_structures = [s.relaxed_atoms if hasattr(s, "relaxed_atoms") else s for s in all_structures]
    logger.info("Loaded %d structures", len(all_structures))

    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    logger.info("Using device: %s", device)

    phase_diagram = loadfn(phase_diagram_path)
    # pourbaix_diagram = loadfn(pourbaix_diagram_path)

    if model_type not in ["DFT"]:
        # Load MACE ensemble calculator
        mace_calc = MACESurface(
            model_paths,
            device=device,
            enable_cueq=True,
        )
    else:
        # Set up compatibility adjustments
        solid_compat = MaterialsProject2020Compatibility()
    if correct_hydroxide_energy:
        if correct_hydrogen_bond_energy:
            hydrogen_bond_correction = DEFAULT_HYDROGEN_BOND_CORRECTION
        else:
            hydrogen_bond_correction = 0.0
        oh_compat = SurfaceOHCompatibility(
            zpe_ts_correction=DEFAULT_OH_ZPE_TS_CORRECTION,
            hydrogen_bond_correction=hydrogen_bond_correction,
        )

    if aq_compat:
        solid_compat = MaterialsProjectAqueousCompatibility(
            solid_compat=solid_compat,
            o2_energy=-4.94795546875,  # DFT energy before any entropy correction
            h2o_energy=-5.192751548333333,  # DFT energy before any entropy correction
            h2o_adjustments=-0.229,  # already counted in the H2O energy
        )

    raw_entries = []
    final_slabs = []
    surf_form_entries = []

    for i, slab in enumerate(all_structures):
        if model_type in ["DFT"]:
            # try to get DFT energies
            try:
                raw_energy = float(slab.info.get("energy", 0)) * HARTREE_TO_EV  # convert to eV
            except (KeyError, AttributeError):
                logger.error("No DFT energy found for %s", slab.get_chemical_formula())
                continue
        else:
            slab.calc = mace_calc
            if relax:
                if i == 0:
                    logger.info("Relaxing the first slab")
                    # save before relaxation
                    slab.write(
                        save_path / f"unrelaxed_{slab.get_chemical_formula()}.cif"
                    )
                slab = optimize_slab(
                    slab,
                    optimizer="FIRE",
                    save_traj=False,
                    relax_steps=relax_steps,
                )[0]
                if i == 0:
                    # save after relaxation
                    slab.write(save_path / f"relaxed_{slab.get_chemical_formula()}.cif")
            results = get_results_single(slab, mace_calc)
            raw_energy = float(results["energy"])  # DFT-like energy

        # Use constraints to set fake surface atoms so that they relax
        if (len(slab.constraints) > 0) and (
            slab.constraints[0].__class__.__name__ == "FixAtoms"
        ):
            fixed_indices = slab.constraints[0].get_indices()
            surface_indices = np.isin(
                np.arange(len(slab)), fixed_indices, invert=True
            ).astype(int)
            slab.set_tags(surface_indices)
        final_slabs.append(slab)
        if input_slab_name:
            slab_name = slab.get_chemical_formula()
        elif input_job_id:
            slab_name = slab.info.get("job_id", None)
        else:
            slab_name = None
        raw_entry = create_computed_entry(slab, raw_energy, slab_name=slab_name)
        print(f"Slab name: {raw_entry.formula}, raw energy: {raw_energy}")

        raw_entries.append(raw_entry)
        if model_type in ["DFT"]:
            # aqcompat.process_entries([raw_entry], inplace=True)  # process the entry
            solid_compat.process_entries([raw_entry], inplace=True)  # process the entry

        if correct_hydroxide_energy:
            oh_compat.process_entries([raw_entry], clean=False, inplace=True)
        print(f"Slab name: {raw_entry.formula}, corrected energy: {raw_entry.energy}")

        # aqcompat.get_adjustments(raw_entry)  #
        # solid_compat.get_adjustments(raw_entry)  #

        surface_formation_entry = create_surface_formation_entry(raw_entry, phase_diagram)
        surf_form_entries.append(surface_formation_entry)

    # Save surface formation entries
    relaxed = "relaxed" if relax else "unrelaxed"
    save_entries_path = (
        save_path / f"{file_base}_{relaxed}_surface_formation_entries_{len(surf_form_entries)}.pkl"
    )
    with open(save_entries_path, "wb") as f:
        pkl.dump(surf_form_entries, f)
    logger.info("Create surface formation entries complete. Saved to %s", save_entries_path)

    # Save final slabs if relaxed
    if relax:
        save_slabs_path = (
            save_path / f"{file_base}_{relaxed}_slab_batches_{len(final_slabs)}.pkl"
        )
        with open(save_slabs_path, "wb") as f:
            pkl.dump(final_slabs, f)
        logger.info(
            "Saved final slabs to %s. Total number of slabs: %d",
            save_slabs_path,
            len(final_slabs),
        )


if __name__ == "__main__":
    args = parse_args()
    main(
        args.surface_name,
        args.file_paths,
        args.model_type,
        args.model_paths,
        args.phase_diagram_path,
        args.pourbaix_diagram_path,
        args.correct_hydroxide_energy,
        args.correct_hydrogen_bond_energy,
        args.aq_compat,
        args.input_slab_name,
        args.input_job_id,
        args.device,
        args.relax,
        args.relax_steps,
        args.save_folder,
    )
