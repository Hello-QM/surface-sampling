"""Active learning loop for iterative MACE fine-tuning with Pourbaix surface sampling.

Implements the cycle:
    Sample surfaces (MCMC) → Cluster for diversity → DFT single-points → Fine-tune MACE → Repeat

Each iteration accumulates training data from all previous iterations so the
fine-tuned model improves monotonically.

Usage:
    # Dry-run (no DFT, uses MACE energies as pseudo-labels):
    python active_learning_loop.py \
        --data_folder data/CuSn_001/ \
        --foundation_model /path/to/MACE-matpes-pbe-omat-ft.model \
        --settings_path scripts/configs/sample_cusn_pourbaix_config.json \
        --num_iterations 3 --skip_dft --device cuda

    # With real DFT:
    python active_learning_loop.py \
        --data_folder data/CuSn_001/ \
        --foundation_model /path/to/MACE-matpes-pbe-omat-ft.model \
        --settings_path scripts/configs/sample_cusn_pourbaix_config.json \
        --num_iterations 3 --dft_command "sbatch vasp.sb {xyz_file}" --device cuda
"""

import argparse
import logging
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from mace.calculators import MACECalculator
from monty.serialization import loadfn
from tqdm import tqdm

from mcmc.calculators import get_embeddings_single
from mcmc.system import SurfaceSystem
from mcmc.utils.clustering import perform_clustering, select_data_and_save
from mcmc.utils.misc import load_dataset_from_files


def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Active learning loop: sample → cluster → DFT → fine-tune → repeat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path to prepared data folder (from prepare_cusn_data.py)",
    )
    p.add_argument(
        "--foundation_model",
        type=str,
        default="/expanse/projects/qstore/csd807/gliu3/ML-potentials/MACE-matpes-pbe-omat-ft.model",
        help="Path to MACE foundation model",
    )
    p.add_argument(
        "--settings_path",
        type=str,
        default="scripts/configs/sample_cusn_pourbaix_config.json",
        help="Path to Pourbaix sampling settings JSON",
    )
    p.add_argument("--num_iterations", type=int, default=3, help="Number of active learning iterations")
    p.add_argument(
        "--structures_per_iter",
        type=int,
        default=20,
        help="Structures to select for DFT per iteration",
    )
    p.add_argument("--clustering_cutoff", type=float, default=200, help="Clustering cutoff (maxclust)")
    p.add_argument("--finetune_epochs", type=int, default=30, help="Fine-tuning epochs per iteration")
    p.add_argument("--finetune_freeze", type=str, default="f5", choices=["f5", "f6"], help="Freeze level")
    p.add_argument(
        "--finetune_lr", type=float, default=1e-3, help="Fine-tuning learning rate"
    )
    p.add_argument(
        "--dft_command",
        type=str,
        default=None,
        help="DFT command template with {xyz_file} placeholder (e.g., 'sbatch vasp.sb {xyz_file}')",
    )
    p.add_argument(
        "--skip_dft",
        action="store_true",
        help="Skip DFT step — use MACE energies as pseudo-labels (for testing)",
    )
    p.add_argument("--save_folder", type=str, default="results/cusn_al/", help="Output directory")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Device")
    p.add_argument(
        "--sampling_sweeps",
        type=int,
        default=None,
        help="Override total_sweeps in settings (for quick testing)",
    )
    p.add_argument(
        "--sampling_sweep_size",
        type=int,
        default=None,
        help="Override sweep_size in settings (for quick testing)",
    )
    return p.parse_args()


def setup_logging(save_folder):
    """Set up logging to file and console."""
    logger = logging.getLogger("active_learning")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(save_folder / "active_learning.log")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# -------------------------------------------------------------------------
# Step 1: Sample surfaces using MCMC
# -------------------------------------------------------------------------
def run_sampling(
    model_path,
    data_folder,
    settings_path,
    iteration,
    save_folder,
    device,
    sampling_sweeps=None,
    sampling_sweep_size=None,
    logger=None,
):
    """Run Pourbaix surface sampling with the current MACE model.

    Args:
        model_path: Path to current MACE model.
        data_folder: Path to data folder with pristine slab and diagrams.
        settings_path: Path to sampling settings JSON.
        iteration: Current iteration number.
        save_folder: Output directory for this iteration.
        device: "cpu" or "cuda".
        sampling_sweeps: Override total_sweeps if provided.
        sampling_sweep_size: Override sweep_size if provided.
        logger: Logger instance.

    Returns:
        list: Sampled structures (SurfaceSystem or Atoms objects).
    """
    from sample_pourbaix_surface import main as sample_main

    data_folder = Path(data_folder)
    iter_folder = save_folder / f"iter_{iteration}" / "sampling"
    iter_folder.mkdir(parents=True, exist_ok=True)

    # Find required files
    slab_files = list(data_folder.glob("*_pristine.pkl"))
    if not slab_files:
        raise FileNotFoundError(f"No pristine slab .pkl found in {data_folder}")
    slab_path = slab_files[0]

    pd_files = list(data_folder.glob("*_pd.json"))
    pbx_files = list(data_folder.glob("*_pbx.json"))
    if not pd_files or not pbx_files:
        raise FileNotFoundError(f"Phase/Pourbaix diagram JSON not found in {data_folder}")

    # Optionally override sampling settings
    settings = loadfn(settings_path)
    if sampling_sweeps is not None:
        settings["sampling_settings"]["total_sweeps"] = sampling_sweeps
    if sampling_sweep_size is not None:
        settings["sampling_settings"]["sweep_size"] = sampling_sweep_size
    settings["sampling_settings"]["run_folder"] = str(iter_folder)

    # Write modified settings to a temp file
    from monty.serialization import dumpfn

    iter_settings_path = iter_folder / "settings.json"
    dumpfn(settings, iter_settings_path)

    logger.info("Iteration %d: Starting MCMC sampling with model %s", iteration, model_path)
    sample_main(
        run_name=f"CuSn_iter{iteration}",
        starting_structure_path=str(slab_path),
        model_path=str(model_path),
        phase_diagram_path=str(pd_files[0]),
        pourbaix_diagram_path=str(pbx_files[0]),
        settings_path=str(iter_settings_path),
        device=device,
        logging_level="info",
    )

    # Load sampled structures
    pkl_files = sorted(iter_folder.glob("*_mcmc_structures.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No MCMC structures found in {iter_folder}")

    with open(pkl_files[-1], "rb") as f:
        structures = pickle.load(f)
    logger.info("Iteration %d: Sampled %d structures", iteration, len(structures))
    return structures


# -------------------------------------------------------------------------
# Step 2: Cluster and select diverse structures
# -------------------------------------------------------------------------
def cluster_and_select(
    structures,
    model_path,
    structures_per_iter,
    clustering_cutoff,
    iteration,
    save_folder,
    device,
    logger=None,
):
    """Cluster sampled structures by MACE embeddings and select diverse representatives.

    Args:
        structures: List of SurfaceSystem or Atoms objects from MCMC.
        model_path: Path to current MACE model for embedding extraction.
        structures_per_iter: Target number of structures to select.
        clustering_cutoff: Cutoff for hierarchical clustering (maxclust).
        iteration: Current iteration number.
        save_folder: Output directory.
        device: "cpu" or "cuda".
        logger: Logger instance.

    Returns:
        list[Atoms]: Selected diverse structures.
    """
    iter_folder = save_folder / f"iter_{iteration}" / "clustering"
    iter_folder.mkdir(parents=True, exist_ok=True)

    # Convert SurfaceSystem → Atoms if needed
    atoms_list = []
    for s in structures:
        if isinstance(s, SurfaceSystem):
            atoms_list.append(s.relaxed_atoms if s.relaxed_atoms is not None else s.real_atoms)
        elif isinstance(s, Atoms):
            atoms_list.append(s)
        else:
            atoms_list.append(s)

    if len(atoms_list) <= structures_per_iter:
        logger.info(
            "Iteration %d: Only %d structures — selecting all (target was %d)",
            iteration,
            len(atoms_list),
            structures_per_iter,
        )
        return atoms_list

    logger.info("Iteration %d: Computing MACE embeddings for %d structures ...", iteration, len(atoms_list))
    device_str = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    single_calc = MACECalculator(model_paths=str(model_path), device=device_str, enable_cueq=True)

    embeddings = []
    energies = []
    for atoms in tqdm(atoms_list, desc=f"Iter {iteration} embeddings"):
        single_calc.calculate(atoms)
        results = single_calc.results
        emb = get_embeddings_single(atoms, single_calc, results_cache=results, flatten=True, flatten_axis=0)
        embeddings.append(emb)
        energies.append(float(results["energy"]))
    embeddings = np.stack(embeddings)
    metric_values = np.array(energies)

    # Hierarchical clustering
    # Use maxclust to get approximately structures_per_iter clusters
    n_clusters = min(structures_per_iter, len(atoms_list))
    logger.info("Iteration %d: Clustering into %d clusters ...", iteration, n_clusters)

    y = perform_clustering(
        embeddings,
        n_clusters,
        "maxclust",
        iter_folder,
        f"iter{iteration}_",
        logger=logger,
    )

    # Select one representative per cluster (lowest energy)
    selected = []
    unique_clusters = np.unique(y)
    for cluster_id in unique_clusters:
        mask = y == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_energies = metric_values[cluster_indices]
        best_idx = cluster_indices[np.argmin(cluster_energies)]
        selected.append(atoms_list[best_idx])

    logger.info("Iteration %d: Selected %d diverse structures", iteration, len(selected))

    # Save selected structures
    with open(iter_folder / f"iter{iteration}_selected_{len(selected)}.pkl", "wb") as f:
        pickle.dump(selected, f)

    return selected


# -------------------------------------------------------------------------
# Step 3: Generate training data (DFT or pseudo-labels)
# -------------------------------------------------------------------------
def generate_training_data(
    selected_structures,
    model_path,
    iteration,
    save_folder,
    device,
    skip_dft=True,
    dft_command=None,
    logger=None,
):
    """Generate training data from selected structures.

    In skip_dft mode, uses the current MACE model to generate pseudo-labels
    (energy + forces). With real DFT, writes input files and collects results.

    Args:
        selected_structures: List of Atoms objects.
        model_path: Path to current MACE model.
        iteration: Current iteration number.
        save_folder: Output directory.
        device: "cpu" or "cuda".
        skip_dft: If True, use MACE pseudo-labels.
        dft_command: DFT command template (required if skip_dft=False).
        logger: Logger instance.

    Returns:
        Path: Path to the generated train.xyz file for this iteration.
    """
    iter_folder = save_folder / f"iter_{iteration}" / "training_data"
    iter_folder.mkdir(parents=True, exist_ok=True)

    train_xyz_path = iter_folder / f"iter{iteration}_train.xyz"

    if skip_dft:
        logger.info("Iteration %d: Generating pseudo-labels with MACE (--skip_dft) ...", iteration)
        device_str = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        calc = MACECalculator(model_paths=str(model_path), device=device_str, enable_cueq=True)

        labeled_atoms = []
        for atoms in tqdm(selected_structures, desc=f"Iter {iteration} pseudo-labels"):
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            energy = atoms_copy.get_potential_energy()
            forces = atoms_copy.get_forces()

            # Store labels in info/arrays for extended XYZ
            atoms_copy.calc = None
            atoms_copy.info["energy"] = float(energy)
            atoms_copy.arrays["forces"] = forces
            labeled_atoms.append(atoms_copy)

        ase_write(str(train_xyz_path), labeled_atoms)
        logger.info("Iteration %d: Wrote %d pseudo-labeled structures to %s",
                     iteration, len(labeled_atoms), train_xyz_path)
    else:
        # Real DFT mode
        if dft_command is None:
            raise ValueError("--dft_command required when --skip_dft is not set")

        # Write structures to individual XYZ files for DFT
        dft_inputs_dir = iter_folder / "dft_inputs"
        dft_inputs_dir.mkdir(exist_ok=True)

        for i, atoms in enumerate(selected_structures):
            xyz_path = dft_inputs_dir / f"structure_{i:04d}.xyz"
            ase_write(str(xyz_path), atoms)

        # Submit DFT jobs
        logger.info(
            "Iteration %d: Submitting %d DFT jobs with command: %s",
            iteration,
            len(selected_structures),
            dft_command,
        )
        for i in range(len(selected_structures)):
            xyz_path = dft_inputs_dir / f"structure_{i:04d}.xyz"
            cmd = dft_command.format(xyz_file=str(xyz_path))
            subprocess.run(cmd, shell=True, check=True)

        logger.info(
            "Iteration %d: DFT jobs submitted. After completion, collect results into %s "
            "in extended XYZ format (energy + forces). Then re-run with --skip_dft "
            "or provide the train.xyz directly.",
            iteration,
            train_xyz_path,
        )
        # Write a placeholder that the user must fill with DFT results
        train_xyz_path.write_text(
            f"# Placeholder: replace with DFT results for iteration {iteration}\n"
            f"# Expected: extended XYZ with energy= and forces per atom\n"
        )

    return train_xyz_path


# -------------------------------------------------------------------------
# Step 4: Fine-tune MACE (head-only)
# -------------------------------------------------------------------------
def finetune_mace(
    train_files,
    foundation_model,
    iteration,
    save_folder,
    freeze_level="f5",
    epochs=30,
    lr=1e-3,
    batch_size=4,
    device="cuda",
    logger=None,
):
    """Fine-tune MACE model using accumulated training data.

    Concatenates training data from all iterations and runs head-only
    fine-tuning on the foundation model (not iteratively on previous fine-tuned
    models, to avoid error accumulation).

    Args:
        train_files: List of paths to train.xyz files from all iterations.
        foundation_model: Path to the original MACE foundation model.
        iteration: Current iteration number.
        save_folder: Output directory.
        freeze_level: "f5" or "f6".
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        device: "cpu" or "cuda".
        logger: Logger instance.

    Returns:
        Path: Path to the fine-tuned model.
    """
    iter_folder = save_folder / f"iter_{iteration}" / "finetune"
    iter_folder.mkdir(parents=True, exist_ok=True)

    # Concatenate all training data from previous iterations
    combined_train_path = iter_folder / "combined_train.xyz"
    all_atoms = []
    for tf in train_files:
        if tf.exists() and tf.stat().st_size > 0:
            try:
                atoms_list = ase_read(str(tf), index=":")
                all_atoms.extend(atoms_list)
            except Exception as e:
                logger.warning("Could not read %s: %s", tf, e)

    if not all_atoms:
        logger.warning("Iteration %d: No training data available, skipping fine-tuning", iteration)
        return Path(foundation_model)

    ase_write(str(combined_train_path), all_atoms)
    logger.info(
        "Iteration %d: Combined %d structures from %d iterations for fine-tuning",
        iteration,
        len(all_atoms),
        len(train_files),
    )

    # Split 80/20 for train/valid
    n_valid = max(1, len(all_atoms) // 5)
    indices = np.random.permutation(len(all_atoms))
    valid_atoms = [all_atoms[i] for i in indices[:n_valid]]
    train_atoms = [all_atoms[i] for i in indices[n_valid:]]

    train_path = iter_folder / "train.xyz"
    valid_path = iter_folder / "valid.xyz"
    ase_write(str(train_path), train_atoms)
    ase_write(str(valid_path), valid_atoms)

    output_model = iter_folder / f"MACE-cusn-iter{iteration}.model"

    logger.info(
        "Iteration %d: Fine-tuning MACE (freeze=%s, epochs=%d, lr=%.1e, train=%d, valid=%d)",
        iteration,
        freeze_level,
        epochs,
        lr,
        len(train_atoms),
        len(valid_atoms),
    )

    # Import and call finetune_headonly.main() logic inline
    # to avoid subprocess overhead and keep everything in one process
    _run_headonly_finetune(
        train_file=str(train_path),
        valid_file=str(valid_path),
        foundation_model=str(foundation_model),
        output=str(output_model),
        freeze_level=freeze_level,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
    )

    if output_model.exists():
        logger.info("Iteration %d: Fine-tuned model saved to %s", iteration, output_model)
        return output_model
    else:
        logger.warning("Iteration %d: Fine-tuning did not produce output, using foundation model", iteration)
        return Path(foundation_model)


def _run_headonly_finetune(
    train_file,
    valid_file,
    foundation_model,
    output,
    freeze_level="f5",
    epochs=30,
    lr=1e-3,
    batch_size=4,
    device="cuda",
    forces_weight=100.0,
    energy_weight=1.0,
    patience=10,
):
    """Run head-only fine-tuning (inlined from finetune_headonly.py).

    This avoids subprocess overhead and keeps everything in-process.
    """
    from mace import data as mace_data
    from mace.tools import torch_geometric, utils

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device_obj = torch.device(device)

    # Load and freeze model
    print(f"Loading foundation model: {foundation_model}")
    model = torch.load(foundation_model, map_location=device_obj)
    if hasattr(model, "model"):
        model = model.model
    if hasattr(model, "models"):
        model = model.models[0]
    model = model.to(device_obj)

    # Freeze all, then selectively unfreeze
    for param in model.parameters():
        param.requires_grad = False

    trainable_prefixes = ["readouts"]
    if freeze_level == "f5":
        trainable_prefixes.append("products")

    for name, param in model.named_parameters():
        for prefix in trainable_prefixes:
            if prefix in name:
                param.requires_grad = True
                break

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Freeze level: {freeze_level}, trainable: {trainable_params:,}/{total_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")

    # Model metadata
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    r_max = float(model.r_max)
    heads = model.heads if hasattr(model, "heads") else ["default"]

    # Load data
    from ase.io import read

    def _load_xyz(filepath):
        atoms_list = read(filepath, index=":")
        for atoms in atoms_list:
            info = atoms.info
            energy = info.get("energy", info.get("REF_energy", info.get("Energy", None)))
            if energy is None and atoms.calc is not None:
                try:
                    energy = atoms.get_potential_energy()
                except Exception:
                    pass
            forces = None
            if "forces" in atoms.arrays:
                forces = atoms.arrays["forces"]
            elif "REF_forces" in atoms.arrays:
                forces = atoms.arrays["REF_forces"]
            elif atoms.calc is not None:
                try:
                    forces = atoms.get_forces()
                except Exception:
                    pass
            if energy is None or forces is None:
                continue
            atoms.calc = None
            atoms.info["energy"] = float(energy)
            atoms.arrays["forces"] = np.array(forces, dtype=np.float64)
        return atoms_list

    def _to_dataset(atoms_list):
        keyspec = mace_data.KeySpecification(
            info_keys={"energy": "energy"},
            arrays_keys={"forces": "forces"},
        )
        dataset = []
        for atoms in atoms_list:
            config = mace_data.config_from_atoms(atoms, key_specification=keyspec)
            atomic_data = mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=r_max, heads=heads)
            dataset.append(atomic_data)
        return dataset

    train_atoms = _load_xyz(train_file)
    train_dataset = _to_dataset(train_atoms)
    train_loader = torch_geometric.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
    )

    valid_loader = None
    if valid_file:
        valid_atoms = _load_xyz(valid_file)
        valid_dataset = _to_dataset(valid_atoms)
        valid_loader = torch_geometric.DataLoader(
            dataset=valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        )

    # Optimizer and scheduler
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience)

    # Training loop
    best_val_loss = float("inf")

    def _compute_loss(out, batch):
        e_true = batch["energy"].to(device_obj)
        f_true = batch["forces"].to(device_obj)
        e_pred = out["energy"]
        f_pred = out["forces"]
        batch_idx = batch["batch"].to(device_obj)
        n_atoms = torch.zeros(e_pred.shape[0], device=device_obj, dtype=e_pred.dtype)
        n_atoms.scatter_add_(0, batch_idx, torch.ones_like(batch_idx, dtype=e_pred.dtype))
        e_loss = torch.mean(((e_pred - e_true) / n_atoms) ** 2)
        f_loss = torch.mean((f_pred - f_true) ** 2)
        return energy_weight * e_loss + forces_weight * f_loss

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device_obj)
            optimizer.zero_grad()
            out = model(batch.to_dict(), training=True)
            loss = _compute_loss(out, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        if valid_loader:
            model.eval()
            val_loss = 0.0
            val_n = 0
            for batch in valid_loader:
                batch = batch.to(device_obj)
                out = model(batch.to_dict(), training=False)
                val_loss += _compute_loss(out, batch).item()
                val_n += 1
            val_loss /= max(val_n, 1)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, output)
        else:
            scheduler.step(train_loss)
            torch.save(model, output)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{epochs}: train_loss={train_loss:.6f}")

    print(f"Fine-tuning complete. Model saved to {output}")


# -------------------------------------------------------------------------
# Main active learning loop
# -------------------------------------------------------------------------
def main():
    args = parse_args()

    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(save_folder)

    logger.info("=" * 70)
    logger.info("Active Learning Loop for CuSn Pourbaix Surface Sampling")
    logger.info("=" * 70)
    logger.info("Foundation model: %s", args.foundation_model)
    logger.info("Data folder: %s", args.data_folder)
    logger.info("Iterations: %d", args.num_iterations)
    logger.info("Structures per iter: %d", args.structures_per_iter)
    logger.info("Skip DFT: %s", args.skip_dft)
    logger.info("Device: %s", args.device)

    current_model = Path(args.foundation_model)
    train_files = []  # Accumulate training data across iterations

    for iteration in range(args.num_iterations):
        logger.info("\n" + "=" * 70)
        logger.info("ITERATION %d / %d", iteration + 1, args.num_iterations)
        logger.info("=" * 70)
        logger.info("Using model: %s", current_model)
        t0 = time.time()

        # Step 1: Sample surfaces
        logger.info("--- Step 1: MCMC Sampling ---")
        structures = run_sampling(
            model_path=current_model,
            data_folder=args.data_folder,
            settings_path=args.settings_path,
            iteration=iteration,
            save_folder=save_folder,
            device=args.device,
            sampling_sweeps=args.sampling_sweeps,
            sampling_sweep_size=args.sampling_sweep_size,
            logger=logger,
        )

        # Step 2: Cluster and select diverse structures
        logger.info("--- Step 2: Clustering & Selection ---")
        selected = cluster_and_select(
            structures=structures,
            model_path=current_model,
            structures_per_iter=args.structures_per_iter,
            clustering_cutoff=args.clustering_cutoff,
            iteration=iteration,
            save_folder=save_folder,
            device=args.device,
            logger=logger,
        )

        # Step 3: Generate training data
        logger.info("--- Step 3: Training Data Generation ---")
        train_xyz = generate_training_data(
            selected_structures=selected,
            model_path=current_model,
            iteration=iteration,
            save_folder=save_folder,
            device=args.device,
            skip_dft=args.skip_dft,
            dft_command=args.dft_command,
            logger=logger,
        )
        train_files.append(train_xyz)

        # Step 4: Fine-tune MACE
        logger.info("--- Step 4: Fine-tuning MACE ---")
        new_model = finetune_mace(
            train_files=train_files,
            foundation_model=args.foundation_model,
            iteration=iteration,
            save_folder=save_folder,
            freeze_level=args.finetune_freeze,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            batch_size=4,
            device=args.device,
            logger=logger,
        )

        # Step 5: Update model for next iteration
        current_model = new_model
        dt = time.time() - t0
        logger.info("Iteration %d complete in %.1f seconds. New model: %s", iteration, dt, current_model)

    logger.info("\n" + "=" * 70)
    logger.info("Active learning loop complete!")
    logger.info("Final model: %s", current_model)
    logger.info("Training data files: %s", [str(f) for f in train_files])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
