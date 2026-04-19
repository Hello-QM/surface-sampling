import numpy as np
import torch
from ase.atoms import Atoms

OUTPUT_KEYS = ["energy", "forces", "embedding"]


def get_mace_prediction(
    calc,
    atoms_list: list[Atoms],
    device: str = "cuda",
    output_keys=OUTPUT_KEYS,
    **kwargs,
) -> dict:
    """Get predictions from a MACE calculator on a list of Atoms.

    Args:
        calc: MACECalculator or MACESurface calculator.
        atoms_list: List of ASE Atoms objects.
        device: Device string (unused, calculator already has device).
        output_keys: Keys to collect from results.

    Returns:
        dict: Collected predictions with keys from output_keys.
    """
    from mcmc.calculators.calculators import get_embeddings_single

    energies = []
    forces_list = []
    embeddings = []

    for atoms in atoms_list:
        calc.calculate(atoms)
        results = calc.results
        energies.append(float(results["energy"]))
        forces_list.append(results["forces"])

        if "embedding" in output_keys:
            emb = get_embeddings_single(atoms, calc, flatten=True)
            embeddings.append(emb)

    predicted = {
        "energy": torch.tensor(energies),
        "forces": torch.tensor(np.concatenate(forces_list, axis=0)),
        "num_atoms": torch.tensor([len(a) for a in atoms_list]),
    }
    if embeddings:
        predicted["embedding"] = torch.tensor(np.stack(embeddings))

    return predicted


# Keep old name as alias
get_nff_prediction = get_mace_prediction


def get_prediction(
    calc,
    dset: list[Atoms],
    batch_size: int = 10,
    device: str = "cuda",
    requires_grad: bool = False,
    **kwargs,
) -> tuple[dict, dict]:
    """Get predictions and targets from a list of Atoms.

    Args:
        calc: MACE calculator.
        dset: List of ASE Atoms objects (with stored energies/forces as targets).
        batch_size: Unused (kept for API compatibility).
        device: Device string.
        requires_grad: Unused.

    Returns:
        tuple[dict, dict]: (target, predicted) dictionaries.
    """
    predicted = get_mace_prediction(calc, dset, device=device, **kwargs)

    target = {"energy": [atoms.get_potential_energy() for atoms in dset]}
    target["energy_grad"] = [-atoms.get_forces(apply_constraint=False) for atoms in dset]

    target["energy_grad"] = np.concatenate(target["energy_grad"], axis=0)

    target["energy"] = torch.tensor(target["energy"]).to(predicted["energy"].device)
    target["forces"] = -torch.tensor(target.pop("energy_grad")).to(predicted["energy"].device)

    return target, predicted


def get_errors(predicted: dict, target: dict, mae=True, rmse=True, r2=True, max_error=True) -> dict:
    pred_energy = predicted["energy"].detach().cpu().numpy()
    targ_energy = target["energy"].detach().cpu().numpy()

    pred_forces = predicted["forces"].detach().cpu().numpy()
    targ_forces = target["forces"].detach().cpu().numpy()
    if pred_energy.ndim > 1 and pred_energy.shape != targ_energy.shape:
        pred_energy = pred_energy.mean(-1)
    if pred_forces.ndim > 2 and pred_forces.shape != targ_forces.shape:
        pred_forces = pred_forces.mean(-1)

    errors = {"energy": {}, "forces": {}}
    if mae:
        mae_energy = np.mean(np.abs(pred_energy - targ_energy))
        mae_forces = np.mean(np.abs(pred_forces - targ_forces))
        errors["energy"]["mae"] = mae_energy
        errors["forces"]["mae"] = mae_forces

    if rmse:
        rmse_energy = np.sqrt(np.mean((pred_energy - targ_energy) ** 2))
        rmse_forces = np.sqrt(np.mean((pred_forces - targ_forces) ** 2))
        errors["energy"]["rmse"] = rmse_energy
        errors["forces"]["rmse"] = rmse_forces

    if r2:
        r2_energy = 1 - np.sum((pred_energy - targ_energy) ** 2) / np.sum(
            (targ_energy - np.mean(targ_energy)) ** 2
        )
        r2_forces = 1 - np.sum((pred_forces - targ_forces) ** 2) / np.sum(
            (targ_forces - np.mean(targ_forces)) ** 2
        )
        errors["energy"]["r2"] = r2_energy
        errors["forces"]["r2"] = r2_forces

    if max_error:
        max_error_energy = np.max(np.abs(pred_energy - targ_energy))
        max_error_forces = np.max(np.abs(pred_forces - targ_forces))
        errors["energy"]["max_error"] = max_error_energy
        errors["forces"]["max_error"] = max_error_forces

    return errors


def get_embedding(
    calc,
    atoms_list: list[Atoms],
    batch_size: int = 10,
    device: str = "cuda",
) -> torch.Tensor:
    """Get embeddings from MACE calculator using forward hooks.

    Args:
        calc: MACE calculator.
        atoms_list: List of ASE Atoms objects.
        batch_size: Unused (kept for API compatibility).
        device: Device string.

    Returns:
        torch.Tensor: Embeddings tensor of shape (n_structures, embedding_dim).
    """
    from mcmc.calculators.calculators import get_embeddings_single

    embeddings = []
    for atoms in atoms_list:
        emb = get_embeddings_single(atoms, calc, flatten=True)
        embeddings.append(emb)

    return torch.tensor(np.stack(embeddings))


def get_prediction_and_errors(
    calc, dset: list[Atoms], batch_size: int = 10, device: str = "cuda"
) -> tuple[dict, dict, dict]:
    """Get predictions, targets, and errors.

    Args:
        calc: MACE calculator.
        dset: List of ASE Atoms objects.
        batch_size: Unused (kept for API compatibility).
        device: Device string.

    Returns:
        tuple[dict, dict, dict]: (target, predicted, errors).
    """
    target, predicted = get_prediction(calc, dset, batch_size, device)

    # Detach tensors
    target = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
    predicted = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in predicted.items()}

    errors = get_errors(predicted, target, mae=True, rmse=True, r2=True, max_error=True)

    return target, predicted, errors


def get_system_val(
    val: list[torch.Tensor], num_atoms: list[torch.Tensor], order: str
) -> torch.Tensor:
    if len(val) == len(num_atoms):
        # It's already a system value
        return val

    splits = torch.split(val, list(num_atoms))
    # Determine the maximum length
    max_length = max(t.size(0) for t in splits)
    padded_tensors = []
    masks = []
    for t in splits:
        padded_length = max_length - t.size(0)
        padded_tensors.append(torch.nn.functional.pad(t, (0, padded_length), "constant", 0))
        # Create a mask that is 1 where data is valid and 0 where it's padded
        mask = torch.ones_like(t, dtype=torch.bool)
        mask = torch.nn.functional.pad(mask, (0, padded_length), "constant", 0)
        masks.append(mask)
    stack_split = torch.stack(padded_tensors, dim=0)
    stacked_masks = torch.stack(masks, dim=0)
    valid_values = stack_split * stacked_masks
    if order == "system_sum":
        system_val = valid_values.sum(dim=-1)
        system_val = system_val.squeeze()
    elif order == "system_max":
        system_val = valid_values.max(dim=-1).values
        system_val = system_val.squeeze()
    elif order == "system_min":
        system_val = valid_values.min(dim=-1).values
        system_val = system_val.squeeze()
    elif order == "system_mean":
        system_val = valid_values.sum(dim=-1) / stacked_masks.sum(dim=-1)
        system_val = system_val.squeeze()
    elif order == "system_mean_squared":
        # valid_values = stack_split * stacked_masks
        system_val = (valid_values**2).sum(dim=-1) / stacked_masks.sum(dim=-1)
        system_val = system_val.squeeze()
    elif order == "system_root_mean_squared":
        # valid_values = stack_split * stacked_masks
        system_val = (valid_values**2).sum(dim=-1) / stacked_masks.sum(dim=-1)
        system_val = system_val.squeeze() ** 0.5
    return system_val


def get_residual(
    targ: dict,
    pred: dict,
    num_atoms: list[int],
    quantity: str = "forces",
    order: str = "system_mean",
) -> torch.Tensor:
    assert pred[quantity].shape == targ[quantity].shape
    # pred[quantity] = pred[quantity].mean(-1)

    res = targ[quantity] - pred[quantity]
    res = abs(res)

    # if quantity == "energy":
    #     return res
    if quantity == "forces" or quantity == "energy_grad":
        res = torch.norm(res, dim=-1)
        if "system" in order:
            system_res = get_system_val(res, num_atoms, order)
            return system_res
    return res
