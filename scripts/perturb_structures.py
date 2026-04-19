"""Perturb structure atomic positions and lattice and save them to a new file."""

import argparse
import pickle as pkl
from datetime import datetime
from logging import getLevelNamesMapping
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from mcmc.calculators import MACESurface
from mcmc.utils import setup_logger
from mcmc.utils.misc import load_dataset_from_files, randomize_structure
from mcmc.utils.plot import plot_surfaces


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Perturb structures and save them to a new file.")
    parser.add_argument(
        "--file_paths",
        nargs="+",
        help="Full paths to pickle or XYZ files of structures",
        type=Path,
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="./",
        help="Folder to save perturbed structures.",
    )
    parser.add_argument(
        "--energy_estimate",
        action="store_true",
        help="Whether to estimate energy before and after perturbation.",
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="*",
        default=[""],
        help="paths to MACE model files",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.3,
        help="Max value of amplitude displacement in Angstroms.",
    )
    parser.add_argument(
        "--displace_lattice",
        action="store_true",
        help="Whether to displace the lattice.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to repeat perturbation.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
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


def main(
    file_names: list[str],
    energy_estimate: bool = False,
    model_paths: list[str] = (""),
    amplitude: float = 0.3,
    displace_lattice: bool = True,
    repeats: int = 1,
    device: Literal["cpu", "cuda"] = "cuda",
    save_folder: Path | str = "./",
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = "info",
) -> None:
    """Perturb structures and save them to a new file.

    Args:
        file_names: List of file paths to load structures from.
        energy_estimate: Whether to estimate energy before and after perturbation.
            Defaults to False.
        model_paths: Paths to MACE model files. Defaults to [""].
        amplitude: Max value of amplitude displacement in Angstroms. Defaults to 0.3.
        displace_lattice: Whether to displace the lattice. Defaults to True.
        repeats: Number of times to repeat perturbation.
        device: Device to use for calculations. Defaults to "cuda".
        save_folder: Folder to save perturbed structures. Defaults to "./".
        logging_level: Logging level. Defaults to "info".
    """
    start_timestamp = datetime.now().isoformat(sep="-", timespec="milliseconds")

    # Initialize save folder
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)
    file_base = f"{start_timestamp}_perturbed_structures"

    # Initialize logger
    logger = setup_logger(
        "perturb_structures",
        save_path / "perturb_structures.log",
        level=getLevelNamesMapping()[logging_level.upper()],
    )

    # Initialize models
    if energy_estimate:
        device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        logger.info("Using device: %s", device)
        mace_calc = MACESurface(
            model_paths,
            device=device,
            enable_cueq=True,
        )

    logger.info("There are a total of %d input files", len(file_names))
    all_structures = load_dataset_from_files(file_names)
    logger.info("Loaded %d structures", len(all_structures))
    all_structures *= repeats
    perturbed_structures = []

    energies_before = []
    energies_after = []
    # Perturb all structures
    for atoms in tqdm(all_structures, desc="Perturbing structures", leave=False):
        # Estimate energy before perturbation
        if energy_estimate:
            logger.info("Estimating energy before perturbation")
            mace_calc.calculate(atoms)
            logger.info("Energy before perturbation: %.3f", float(mace_calc.results["energy"]))
            energies_before.append(float(mace_calc.results["energy"]))
        perturbed_atoms = randomize_structure(atoms, amplitude, displace_lattice=displace_lattice)
        # Estimate energy after perturbation
        if energy_estimate:
            logger.info("Estimating energy after perturbation")
            mace_calc.calculate(perturbed_atoms)
            logger.info("Energy after perturbation: %.3f", float(mace_calc.results["energy"]))
            energies_after.append(float(mace_calc.results["energy"]))
        perturbed_structures.append(perturbed_atoms)

    if energy_estimate:
        average_diff = np.mean(np.array(energies_after) - np.array(energies_before))
        logger.info("Average energy difference: %.3f eV", average_diff)

    # Plot 5 sampled structures, before and after perturbation
    sampled_idx = np.random.choice(len(all_structures), 5, replace=False)
    logger.info("Sampling before and after structures at indices: %s", sampled_idx)
    plot_surfaces(
        [all_structures[x] for x in sampled_idx] + [perturbed_structures[x] for x in sampled_idx],
        fig_name=file_base,
        save_folder=save_path,
    )

    # Save cif samples
    for i, idx in enumerate(sampled_idx):
        atoms = all_structures[idx]
        atoms.write(save_path / f"{file_base}_sample_{i}_before.cif")

        atoms = perturbed_structures[idx]
        atoms.write(save_path / f"{file_base}_sample_{i}_after.cif")

    # Save perturbed structures
    perturbed_surface_path = (
        save_path / f"{file_base}_total_{len(perturbed_structures)}_amp_{amplitude}_structures.pkl"
    )
    with open(
        perturbed_surface_path,
        "wb",
    ) as f:
        pkl.dump(perturbed_structures, f)

    logger.info("Structure perturbation complete. Saved to %s", perturbed_surface_path)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.file_paths,
        args.energy_estimate,
        args.model_paths,
        args.amplitude,
        args.displace_lattice,
        args.repeats,
        device=args.device,
        save_folder=args.save_folder,
        logging_level=args.logging_level,
    )
