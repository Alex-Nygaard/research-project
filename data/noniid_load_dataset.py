"""Partition the data and create the dataloaders."""

from typing import List
import torch
from torch.utils.data import DataLoader, random_split, Subset

from .noniid_dataset_preparation import (
    get_dirichlet_info,
    get_custom_info,
    _download_data,
)


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


def load_validation_set(
    dataset_name: str = "cifar10",
    download: bool = False,
):
    _, testset = _download_data(dataset_name, download=download)
    return DataLoader(testset, batch_size=32)
