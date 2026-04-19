"""Miscellaneous utility functions for the MCMC workflow."""

import pickle as pkl
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from ase import io
from ase.atoms import Atoms
from scipy.spatial import distance
from scipy.special import softmax
from tqdm import tqdm


def get_atoms_batch(
    data: dict | Atoms,
    nff_cutoff: float = 6.0,
    device: str = "cpu",
    **kwargs,
) -> Atoms:
    """Return an ASE Atoms object from atoms or dictionary.

    MACE handles neighbor lists internally, so no AtomsBatch wrapping is needed.
    This function is kept for API compatibility.

    Args:
        data (Union[dict, Atoms]): Dictionary or ASE Atoms containing the properties of the atoms
        nff_cutoff (float): Unused (kept for API compatibility). Defaults to 6.0.
        device (str, optional): Unused (kept for API compatibility). Defaults to 'cpu'.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Atoms: Plain ASE Atoms object.
    """
    if isinstance(data, Atoms):
        return data
    # dict case: reconstruct Atoms from dictionary
    return Atoms.fromdict(data)


def get_atoms_batches(
    data: list[Atoms],
    nff_cutoff: float = 6.0,
    device: str = "cpu",
    structures_per_batch: int = 32,
    **kwargs,
) -> list[Atoms]:
    """Return a list of ASE Atoms objects.

    MACE handles neighbor lists internally, so no AtomsBatch wrapping is needed.
    This function is kept for API compatibility.

    Args:
        data (list[ase.Atoms]): List of ASE Atoms objects.
        nff_cutoff (float): Unused (kept for API compatibility). Defaults to 6.0.
        device (str, optional): Unused (kept for API compatibility). Defaults to 'cpu'.
        structures_per_batch (int, optional): Unused. Defaults to 32.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        list[Atoms]: List of ASE Atoms objects.
    """
    print(f"Data has length {len(data)}")
    return list(data)


def load_dataset_from_files(file_paths: list[Path | str]) -> list[Atoms]:
    """Load dataset from files. Dataset can be a list of ASE Atoms objects or pickle files.

    Args:
        file_paths (list[Path]): List of file paths.

    Returns:
        list[Atoms]: List of ASE Atoms objects.
    """
    file_paths = [Path(file_name) for file_name in file_paths]

    dset = []
    for x in file_paths:
        if x.suffix == ".txt":
            # load file paths from a text file
            with open(x, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return load_dataset_from_files([Path(line.strip()) for line in lines])
        elif x.suffix == ".pkl":
            with open(x, "rb") as f:
                dset.extend(pkl.load(f))
        elif x.suffix == ".xyz":
            atoms_list = io.read(str(x), index=":")
            dset.extend(atoms_list)
        else:
            # Try pickle for .pth.tar and other formats
            with open(x, "rb") as f:
                data = pkl.load(f)
            if isinstance(data, list):
                dset.extend(data)
            else:
                dset.append(data)
    return dset


def filter_distances(slab: Atoms, ads: Iterable = ("O"), cutoff_distance: float = 1.5) -> bool:
    """This function filters out slabs that have atoms too close to each other based on a
    specified cutoff distance.

    Args:
        slab (Atoms): The slab structure to check for distances.
        ads (Iterable, optional): The adsorbate atom types in the slab to check for. Defaults to
            ("O").
        cutoff_distance (float, optional): The cutoff distance to check for. Defaults to 1.5.

    Returns:
        bool: True if the distances are greater than the cutoff distance, False otherwise.
    """
    # Checks distances of all adsorbates are greater than cutoff
    ads_arr = np.isin(slab.get_chemical_symbols(), ads)
    unique_dists = np.unique(np.triu(slab.get_all_distances(mic=True)[ads_arr][:, ads_arr]))
    # Get upper triangular matrix of ads dist
    return not any(unique_dists[(unique_dists > 0) & (unique_dists <= cutoff_distance)])


def randomize_structure(atoms, amplitude, displace_lattice=True) -> Atoms:
    """Randomly displaces the atomic coordinates (and lattice parameters)
    by a certain amplitude. Useful to generate slightly off-equilibrium
    configurations and starting points for MD simulations. The random
    amplitude is sampled from a uniform distribution.

    Same function as in pymatgen, but for ase.Atoms objects.

    Args:
        atoms (ase.Atoms): The input structure.
        amplitude (float): Max value of amplitude displacement in Angstroms.
        displace_lattice (bool): Whether to displace the lattice.

    Returns:
        ase.Atoms: The perturbed structure.
    """
    newcoords = atoms.get_positions() + np.random.uniform(
        -amplitude, amplitude, size=atoms.positions.shape
    )

    newlattice = np.array(atoms.get_cell())
    if displace_lattice:
        newlattice += np.random.uniform(-amplitude, amplitude, size=newlattice.shape)

    return Atoms(
        positions=newcoords,
        numbers=atoms.numbers,
        cell=newlattice,
        pbc=atoms.pbc,
    )


def compute_distance_weight_matrix(
    ads_coords: np.ndarray, distance_decay_factor: float
) -> np.ndarray:
    """Compute distance weight matrix using softmax.

    Args:
        ads_coords (np.ndarray): The coordinates of the adsorption sites.
        distance_decay_factor (float): Exponential decay factor.

    Returns:
        np.ndarray: The distance weight matrix.
    """
    # Compute pairwise distance matrix
    ads_coord_distances = distance.cdist(ads_coords, ads_coords, "euclidean")

    # Compute distance decay matrix using softmax
    distance_weight_matrix = softmax(-ads_coord_distances / distance_decay_factor, axis=1)

    assert np.allclose(np.sum(distance_weight_matrix, axis=1), 1.0)

    return distance_weight_matrix
