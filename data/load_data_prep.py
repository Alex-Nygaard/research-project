import os
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


def replicate_client_distributions(
    num_clients: int,
    datapoints_per_client: List[int],
    labels_per_client: List[int],
    seed=42,
    download=False,
):
    """
    Get the indices of the data points for each client by replicating the distribution of the original dataset.

    :param num_clients: Number of clients.
    :param datapoints_per_client: Number of data points per client.
    :param labels_per_client: Number of labels per client.
    :param seed: Seed for the random number generator.
    :param download: If True, download the dataset.
    :return: Tuple of data indices, data volumes, and data labels.
    """

    assert (
        len(datapoints_per_client) == num_clients
    ), "The length of datapoints_per_client must be equal to num_clients."
    assert (
        len(labels_per_client) == num_clients
    ), "The length of labels_per_client must be equal to num_clients."

    trainset, testset = get_cifar(download=download)
    prng = np.random.default_rng(seed)

    targets = trainset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    num_classes = len(set(targets))

    # Create a mapping from each label to the corresponding indices
    label_indices = {k: np.where(targets == k)[0].tolist() for k in range(num_classes)}
    for indices in label_indices.values():
        prng.shuffle(indices)

    # Initialize client indices
    idx_clients: List[List[int]] = [[] for _ in range(num_clients)]

    for client_id, (num_samples, num_labels) in enumerate(
        zip(datapoints_per_client, labels_per_client)
    ):
        selected_labels = prng.choice(range(num_classes), num_labels, replace=False)
        client_indices = []

        for label in selected_labels:
            count_needed = (num_samples // num_labels) + (
                1 if num_samples % num_labels > 0 else 0
            )
            count_assigned = 0

            while label_indices[label] and count_assigned < count_needed:
                client_indices.append(label_indices[label].pop())
                count_assigned += 1

        prng.shuffle(client_indices)
        idx_clients[client_id] = client_indices[:num_samples]

    # Ensure all datapoints are used
    remaining_indices = [idx for indices in label_indices.values() for idx in indices]
    prng.shuffle(remaining_indices)

    for client_id in range(num_clients):
        if len(idx_clients[client_id]) < datapoints_per_client[client_id]:
            required = datapoints_per_client[client_id] - len(idx_clients[client_id])
            idx_clients[client_id].extend(remaining_indices[:required])
            remaining_indices = remaining_indices[required:]

    data_volume = [len(idxs) for idxs in idx_clients]
    data_labels = [
        len(set([ex[1] for ex in [trainset[idx] for idx in idxs]]))
        for idxs in idx_clients
    ]

    return idx_clients, data_volume, data_labels


def dirichlet_distributions(num_clients, alpha, seed=42, download=True):
    """
    Get the indices of the data points for each client using a Dirichlet distribution.

    Adapted from https://github.com/adap/flower/blob/main/baselines/niid_bench/niid_bench/dataset_preparation.py

    :param num_clients: Number of clients.
    :param alpha: Alpha parameter of the Dirichlet distribution.
    :param seed: Seed for the random number generator.
    :param download: If True, download the dataset.
    :return: Tuple of data indices, data volumes, and data labels.
    """
    trainset, testset = get_cifar(download=download)
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(seed)

    tmp_t = trainset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    while min_samples < min_required_samples_per_client:
        idx_clients: List[List] = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    p * (len(idx_j) < total_samples / num_clients)
                    for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

    data_volume = [len(idx_j) for idx_j in idx_clients]
    data_labels = [
        len(set([ex[1] for ex in [trainset[idx] for idx in idxs]]))
        for idxs in idx_clients
    ]
    return idx_clients, data_volume, data_labels


def get_cifar(download=True) -> Tuple[Dataset, Dataset]:
    """
    Load the CIFAR10 dataset.

    Adapted from https://github.com/adap/flower/blob/main/baselines/niid_bench/niid_bench/dataset_preparation.py

    :param download: If True, download the dataset.
    :return: Tuple of train and test datasets.
    """
    trainset, testset = None, None

    root = "data"
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    if slurm_job_id is not None:
        root = f"/tmp/{slurm_job_id}/data"
        os.makedirs(root, exist_ok=True)

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4),
                    mode="reflect",
                ).data.squeeze()
            ),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    trainset = CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=transform_train,
    )
    testset = CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transform_test,
    )

    return trainset, testset
