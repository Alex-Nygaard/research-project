"""Partition the data and create the dataloaders."""

from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from .noniid_dataset_preparation import (
    partition_data,
    partition_data_dirichlet,
    partition_data_label_quantity,
    partition_data_counts,
    partition_data_varying_labels,
    partition_data_varying_labels_equal_num,
    partition_data_custom,
)


# pylint: disable=too-many-locals, too-many-branches
def load_datasets(
    config,
    num_clients: int,
    val_ratio: float = 0.1,
    seed: Optional[int] = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: dict
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoaders for training, validation, and testing.
    """
    print(f"Dataset partitioning config: {config}")
    partitioning = ""
    if "partitioning" in config:
        partitioning = config["partitioning"]
    # partition the data
    if partitioning == "dirichlet":
        alpha = 0.5
        if "alpha" in config:
            alpha = config["alpha"]
        datasets, testset = partition_data_dirichlet(
            num_clients,
            alpha=alpha,
            seed=seed,
            dataset_name=config["name"],
        )
    elif partitioning == "label_quantity":
        labels_per_client = 2
        if "labels_per_client" in config:
            labels_per_client = config["labels_per_client"]
        datasets, testset = partition_data_label_quantity(
            num_clients,
            labels_per_client=labels_per_client,
            seed=seed,
            dataset_name=config["name"],
        )
    elif partitioning == "iid":
        datasets, testset = partition_data(
            num_clients,
            similarity=1.0,
            seed=seed,
            dataset_name=config["name"],
        )
    elif partitioning == "iid_noniid":
        similarity = 0.1
        if "similarity" in config:
            similarity = config["similarity"]
        datasets, testset = partition_data(
            num_clients,
            similarity=similarity,
            seed=seed,
            dataset_name=config["name"],
        )
    elif partitioning == "counts":
        counts = config["counts"]
        datasets, testset = partition_data_counts(
            counts, seed=seed, dataset_name=config["name"]
        )
    elif partitioning == "labels_noniid":
        datasets, testset = partition_data_varying_labels(
            num_clients,
            seed=seed,
            dataset_name=config["name"],
        )
    elif partitioning == "labels_noniid_equal":
        datasets, testset = partition_data_varying_labels_equal_num(
            num_clients,
            seed=seed,
            dataset_name=config["name"],
        )
    elif partitioning == "replicate":
        datasets, testset = partition_data_custom(
            num_clients,
            datapoints_per_client=config["datapoints_per_client"],
            labels_per_client=config["labels_per_client"],
            seed=seed,
            dataset_name=config["name"],
        )
    else:
        raise ValueError("something")

    batch_sizes: List[int] = config["batch_sizes"]

    assert len(batch_sizes) == len(
        datasets
    ), "Batch sizes and datasets must have the same lengths"

    # split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for batch_size, dataset in zip(batch_sizes, datasets):
        len_val = int(len(dataset) / (1 / val_ratio)) if val_ratio > 0 else 0
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return (
        trainloaders,
        valloaders,
        DataLoader(testset, batch_size=len(testset)),
    )
