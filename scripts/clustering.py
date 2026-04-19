"""Clustering of structures based on their latent space embeddings."""

import argparse
from datetime import datetime
from logging import getLevelNamesMapping
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from mace.calculators import MACECalculator
from tqdm import tqdm

from mcmc.calculators import MACESurface, get_embeddings_single, get_results_single, get_std_devs_single
from mcmc.system import SurfaceSystem
from mcmc.uncertainty import Uncertainty
from mcmc.utils import setup_logger
from mcmc.utils.clustering import perform_clustering, select_data_and_save
from mcmc.utils.misc import load_dataset_from_files

np.set_printoptions(precision=3, suppress=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cluster structures based on their latent space embeddings."
    )
    parser.add_argument(
        "--file_paths",
        nargs="+",
        help="Full paths to pickle files or XYZ files of structures.",
        type=Path,
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="./",
        help="Folder to save cut surfaces.",
    )
    parser.add_argument(
        "--model_paths",
        nargs="*",
        help="Full paths to MACE model files",
        type=str,
        default=[""],
    )
    parser.add_argument(
        "--max_input_len",
        help="Maximum number of structures used in each clustering iteration",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--clustering_metric",
        help="Metric used to select structure from each cluster",
        choices=("force_std", "random", "energy", "gmm"),
        type=str,
        default="force_std",
    )
    parser.add_argument(
        "--gmm_path",
        help="Full path to GMM model",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--cutoff_criterion",
        choices=("distance", "maxclust"),
        help="Cutoff criterion, either distance or maxclust",
        type=str,
        default="distance",
    )
    parser.add_argument(
        "--clustering_cutoff",
        help=(
            "Clustering cutoff, either the cutoff distance between surfaces "
            "or the maximum number of clusters"
        ),
        type=float,
        default=200,
    )
    parser.add_argument(
        "--device",
        help="Device, either cpu or cuda",
        type=str,
        default="cuda",
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
    device: str = "cuda",
    clustering_cutoff: float = 0.2,
    cutoff_criterion: Literal["distance", "maxclust"] = "distance",
    clustering_metric: Literal["force_std", "random", "energy", "gmm"] = "force_std",
    gmm_path: Path | str | None = None,
    max_input_len: int = 1000,
    model_paths: list[str] | None = None,
    save_folder: Path | str = "./",
    logging_level: Literal["debug", "info", "warning", "error", "critical"] = "info",
) -> None:
    """Main function to perform clustering on a list of structures

    Args:
        file_names (list[str]) : List of file paths to load structures from
        device (str, optional) : cpu or cuda device, by default 'cuda'
        clustering_cutoff (float, optional) : Either the distance or the maximum number of clusters,
            by default 0.2
        max_input_len (int, optional) : Maximum number of structures used in each clustering
            iteration, by default 1000
        clustering_metric (Literal['force_std', 'random', 'energy', 'gmm'], optional) : Metric used
            to select structure from each cluster, by default 'force_std'
        gmm_path (Path | str, optional) : Full path to GMM model, by default None
        model_paths (list[str], optional) : Full paths to MACE model files, by default None
        cutoff_criterion (Literal['distance', 'maxclust'], optional) : Either distance or maxclust,
            by default 'distance'
        save_folder (Union[Path, str], optional) : Folder to save the plots, by default "./"
        logging_level (Literal['debug', 'info', 'warning', 'error', 'critical'], optional) : Logging
            level, by default 'info'
    """
    start_timestamp = datetime.now().isoformat(sep="-", timespec="milliseconds")

    # Initialize run folder
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = setup_logger(
        "clustering",
        save_path / "clustering.log",
        level=getLevelNamesMapping()[logging_level.upper()],
    )

    logger.info("There are a total of %d input files", len(file_names))
    dset = load_dataset_from_files(file_names)
    logger.info("Loaded %d structures", len(dset))

    if isinstance(dset[0], SurfaceSystem):
        logger.info("Loaded SurfaceSystem object")
        dset = [system.relaxed_atoms for system in dset]
        logger.info("Converted to list of Atoms objects")

    device = "cuda" if torch.cuda.is_available() and "cpu" not in device else "cpu"
    logger.info("Using %s device for MACE calculations", device)

    if model_paths:
        logger.info("Loading MACE models from %s", model_paths)
    else:
        raise ValueError("No MACE models provided")

    # MACESurface for ensemble force standard deviation
    ensemble_calc = MACESurface(
        model_paths,
        device=device,
        enable_cueq=True,
    )
    # Single MACECalculator for embedding extraction
    single_calc = MACECalculator(
        model_paths=model_paths[0],
        device=device,
        enable_cueq=True,
    )

    # Load GMM if clustering metric is gmm
    if clustering_metric == "gmm":
        gmm_model = Uncertainty.load(gmm_path)

    # Perform clustering in batches
    num_batches = len(dset) // max_input_len + bool(
        len(dset) % max_input_len
    )  # additional batch for the remainder

    logger.info("Performing clustering in %d batches", num_batches)
    for i in range(num_batches):
        dset_batch = (
            dset[i * max_input_len : (i + 1) * max_input_len]
            if i < num_batches - 1
            else dset[i * max_input_len :]
        )
        batch_number = i + 1
        logger.info("Starting clustering for batch # %d", batch_number)

        file_base = f"{start_timestamp}_clustering"
        save_prepend = (
            file_base
            + f"_{len(dset_batch)}_input_structures"
            + f"_batch_{batch_number}".zfill(3)
            + f"_cutoff_{clustering_cutoff}_"
            + f"{clustering_metric}_"
        )

        # doing it singly to save memory and is faster
        embeddings = []
        metric_values = []
        for single_dset in tqdm(dset_batch):
            single_calc.calculate(single_dset)
            single_calc_results = single_calc.results
            embedding = get_embeddings_single(
                single_dset,
                single_calc,
                results_cache=single_calc_results,
                flatten=True,
                flatten_axis=0,
            )
            if clustering_metric == "energy":
                metric_value = float(single_calc_results["energy"])
            elif clustering_metric == "force_std":
                metric_value = get_std_devs_single(single_dset, ensemble_calc)
            elif clustering_metric == "gmm":
                # GMM on embedding
                emb_tensor = torch.tensor(embedding).unsqueeze(0)
                metric_value = float(gmm_model(
                    {"embedding": emb_tensor},
                    num_atoms=torch.tensor([len(single_dset)])
                ).item())
            else:
                metric_value = np.random.rand()
            embeddings.append(embedding)
            metric_values.append(metric_value)
        embeddings = np.stack(embeddings)
        metric_values = np.stack(metric_values)

        y = perform_clustering(
            embeddings, clustering_cutoff, cutoff_criterion, save_path, save_prepend, logger=logger
        )
        select_data_and_save(
            dset_batch, y, metric_values, clustering_metric, save_path, save_prepend, logger=logger
        )
    logger.info("Clustering complete!")


if __name__ == "__main__":
    args = parse_args()
    main(
        args.file_paths,
        device=args.device,
        clustering_cutoff=args.clustering_cutoff,
        cutoff_criterion=args.cutoff_criterion,
        clustering_metric=args.clustering_metric,
        gmm_path=args.gmm_path,
        max_input_len=args.max_input_len,
        model_paths=args.model_paths,
        save_folder=args.save_folder,
        logging_level=args.logging_level,
    )
