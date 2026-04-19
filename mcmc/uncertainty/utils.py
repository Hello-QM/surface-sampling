import ase
import numpy as np
from ase import Atoms

HARTREE_TO_EV = 27.211386245988


def make_uncertainty_dataset(atoms_list: list[Atoms], cutoff=6.0) -> list[Atoms]:
    """Prepare atoms for uncertainty calculations.

    MACE works directly with ASE Atoms, so no special dataset wrapping is needed.

    Args:
        atoms_list: List of ASE Atoms objects.
        cutoff: Unused (kept for API compatibility).

    Returns:
        list[Atoms]: The same list of Atoms objects.
    """
    return list(atoms_list)


def shrink_uncertainty_dataset(dataset: list[Atoms], indices) -> list[Atoms]:
    """Select a subset of atoms from the dataset by indices.

    Args:
        dataset: List of ASE Atoms objects.
        indices: Indices to select.

    Returns:
        list[Atoms]: Selected subset.
    """
    return [dataset[idx] for idx in indices]


def make_clustering_dataset(
    atoms_list: list[Atoms], center_atom_index_list: list[int], cutoff: float = 6.0
) -> list[Atoms]:
    """Prepare atoms for clustering with center atom indices stored in atoms.info.

    Args:
        atoms_list: List of ASE Atoms objects.
        center_atom_index_list: List of center atom indices for each structure.
        cutoff: Unused (kept for API compatibility).

    Returns:
        list[Atoms]: Atoms with center_idx stored in atoms.info.
    """
    result = []
    for i, atoms in enumerate(atoms_list):
        atoms_copy = atoms.copy()
        atoms_copy.info["center_idx"] = center_atom_index_list[i]
        result.append(atoms_copy)
    return result


def preprocess_traj(
    total_candidates: list[ase.Atoms], z_cutoff: int | None = None, z_threshold: float | None = None
) -> list[ase.Atoms]:
    new_total_candidates = []
    for atoms in total_candidates:
        sorted_atoms = sorted(atoms, key=lambda atom: atom.position[2])
        sorted_atoms = ase.Atoms(sorted_atoms, pbc=atoms.pbc, cell=atoms.cell)
        new_pos = []
        if z_cutoff is not None:
            if z_threshold is None:
                z_threshold = 0.1
            z_coords_with_indices = sorted(
                (z, i) for i, z in enumerate(atoms.get_positions()[:, 2])
            )
            z_coords, indices = group_layers_with_indices(z_coords_with_indices, z_threshold)
            removing_indices = []
            for i in range(z_cutoff):
                removing_indices += indices[i]
            # print(np.array(z_coords[0]))
            shift_val = np.mean(np.array(z_coords[z_cutoff])) - np.mean(np.array(z_coords[0]))
            reduced_atoms = atoms.copy()

            del reduced_atoms[removing_indices]
            for pos in reduced_atoms.get_positions():
                new_pos.append(pos + np.array([0, 0, -shift_val]))
            reduced_atoms.set_positions(new_pos)
            sorted_atoms = reduced_atoms.copy()
        new_total_candidates.append(sorted_atoms)
    return new_total_candidates


def group_layers_with_indices(z_coords, threshold):
    layers = []
    indices = []
    current_layer = [z_coords[0][0]]
    current_indices = [z_coords[0][1]]

    for z, i in z_coords[1:]:
        if z - current_layer[-1] <= threshold:
            current_layer.append(z)
            current_indices.append(i)
        else:
            layers.append(current_layer)
            indices.append(current_indices)
            current_layer = [z]
            current_indices = [i]
    layers.append(current_layer)  # Append the last layer
    indices.append(current_indices)  # Append the last indices

    return layers, indices
