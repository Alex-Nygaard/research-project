"""Partition the data and create the dataloaders."""

from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset

from .noniid_dataset_preparation import (
    # partition_data,
    partition_data_dirichlet,
    partition_data_custom,
    get_dirichlet_info,
    get_custom_info,
    _download_data,
    # partition_data_label_quantity,
    # partition_data_counts,
    # partition_data_varying_labels,
    # partition_data_varying_labels_equal_num,
)


def load_datasets_with_idxs(name: str, train: bool = True):
    pass


def get_dirichlet_idxs(
    num_clients: int,
    alpha: float = 0.5,
    val_ratio: float = 0.1,
    seed: int = 42,
    download: bool = True,
):
    idxs, data_volume, data_labels = get_dirichlet_info(
        num_clients, alpha=alpha, seed=seed, download=download
    )

    return idxs, data_volume, data_labels


def get_replicate_idxs(
    num_clients: int,
    datapoints_per_client: List[int],
    labels_per_client: List[int],
    val_ratio: float = 0.1,
    seed: int = 42,
    dowlnoad: bool = True,
):
    idxs, data_volume, data_labels = get_custom_info(
        num_clients,
        datapoints_per_client=datapoints_per_client,
        labels_per_client=labels_per_client,
        seed=seed,
        download=dowlnoad,
    )

    return idxs, data_volume, data_labels


def load_one_client(
    idx: List[int],
    batch_size: int,
    dataset_name: str = "cifar10",
    download: bool = False,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    trainset, testset = _download_data(dataset_name, download=download)

    dataset = Subset(trainset, idx)

    len_val = int(len(dataset) / (1 / val_ratio)) if val_ratio > 0 else 0
    lengths = [len(dataset) - len_val, len_val]
    ds_train, ds_val = random_split(
        dataset, lengths, torch.Generator().manual_seed(seed)
    )
    return (
        DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(ds_val, batch_size=batch_size),
    )


def load_all_clients(
    idx_clients: List[List[int]],
    batch_sizes: List[int],
    dataset_name: str,
    download: bool = False,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    trainset, testset = _download_data(dataset_name, download=download)

    datasets = [Subset(trainset, idxs) for idxs in idx_clients]

    # split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for batch_size, dataset in zip(batch_sizes, datasets):
        len_val = int(len(dataset) / (1 / val_ratio)) if val_ratio > 0 else 0
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(
            DataLoader(
                ds_train,
                batch_size=batch_size,
                shuffle=True,
            )
        )
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return (
        trainloaders,
        valloaders,
    )


def load_validation_set(
    dataset_name: str = "cifar10",
    download: bool = False,
):
    trainset, testset = _download_data(dataset_name, download=download)
    return DataLoader(testset, batch_size=64)


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
        trainloaders.append(
            DataLoader(
                ds_train,
                batch_size=batch_size,
                shuffle=True,
            )
        )
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return (
        trainloaders,
        valloaders,
        DataLoader(testset, batch_size=len(testset)),
    )
