"""Download data and partition data with different partitioning strategies."""

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

# def partition_data_counts(
#     client_data_counts: List[int], seed=42, dataset_name="cifar10"
# ) -> Tuple[List[Dataset], Dataset]:
#     trainset, testset = _download_data(dataset_name)
#     prng = np.random.default_rng(seed)
#
#     cumulative = 0
#     trainset_per_client = []
#     for count in client_data_counts:
#         trainset_per_client.append(
#             Subset(trainset, range(cumulative, cumulative + count))
#         )
#         cumulative += count
#
#     return trainset_per_client, testset
#
#
# def partition_data_varying_labels(
#     num_clients,
#     mean_labels=5,
#     std_labels=2,
#     min_labels=1,
#     max_labels=10,
#     seed=42,
#     dataset_name="cifar10",
# ) -> Tuple[List[Dataset], Dataset]:
#     """Partition the data to give all clients an equal amount of datapoints, but with a varying amount of labels.
#
#     Parameters
#     ----------
#     num_clients : int
#         The number of clients that hold a part of the data.
#     mean_labels : float, optional
#         The mean of the normal distribution for the number of labels per client, by default 5.
#     std_labels : float, optional
#         The standard deviation of the normal distribution for the number of labels per client, by default 2.
#     min_labels : int, optional
#         The minimum number of labels per client, by default 1.
#     max_labels : int, optional
#         The maximum number of labels per client, by default 10.
#     seed : int, optional
#         Used to set a fixed seed to replicate experiments, by default 42.
#     dataset_name : str
#         Name of the dataset to be used.
#
#     Returns
#     -------
#     Tuple[List[Subset], Dataset]
#         The list of datasets for each client, the test dataset.
#     """
#     trainset, testset = _download_data(dataset_name)
#     prng = np.random.default_rng(seed)
#
#     targets = trainset.targets
#     if isinstance(targets, list):
#         targets = np.array(targets)
#     if isinstance(targets, torch.Tensor):
#         targets = targets.numpy()
#     num_classes = len(set(targets))
#     num_samples_per_client = len(trainset) // num_clients
#
#     # Generate the number of labels for each client
#     labels_per_client = np.clip(
#         prng.normal(loc=mean_labels, scale=std_labels, size=num_clients).astype(int),
#         min_labels,
#         max_labels,
#     )
#
#     times = [0 for _ in range(num_classes)]
#     contains = []
#
#     for i in range(num_clients):
#         current = [i % num_classes]
#         times[i % num_classes] += 1
#         j = 1
#         while j < labels_per_client[i]:
#             index = prng.choice(num_classes, 1)[0]
#             if index not in current:
#                 current.append(index)
#                 times[index] += 1
#                 j += 1
#         contains.append(current)
#
#     idx_clients: List[List] = [[] for _ in range(num_clients)]
#     for i in range(num_classes):
#         idx_k = np.where(targets == i)[0]
#         prng.shuffle(idx_k)
#         idx_k_split = np.array_split(idx_k, times[i])
#         ids = 0
#         for j in range(num_clients):
#             if i in contains[j]:
#                 idx_clients[j] += idx_k_split[ids].tolist()
#                 ids += 1
#
#     # Ensure each client has the same number of datapoints
#     trainsets_per_client = []
#     for idxs in idx_clients:
#         if len(idxs) > num_samples_per_client:
#             idxs = prng.choice(idxs, num_samples_per_client, replace=False).tolist()
#         elif len(idxs) < num_samples_per_client:
#             idxs = np.random.choice(idxs, num_samples_per_client, replace=True).tolist()
#         trainsets_per_client.append(Subset(trainset, idxs))
#
#     return trainsets_per_client, testset
#
#
# def partition_data_varying_labels_equal_num(
#     num_clients,
#     mean_labels=5,
#     std_labels=2,
#     min_labels=1,
#     max_labels=10,
#     seed=42,
#     dataset_name="cifar10",
# ) -> Tuple[List[Dataset], Dataset]:
#     """Partition the data to give all clients an equal amount of datapoints, but with a varying amount of labels.
#
#     Parameters
#     ----------
#     num_clients : int
#         The number of clients that hold a part of the data.
#     mean_labels : float, optional
#         The mean of the normal distribution for the number of labels per client, by default 5.
#     std_labels : float, optional
#         The standard deviation of the normal distribution for the number of labels per client, by default 2.
#     min_labels : int, optional
#         The minimum number of labels per client, by default 1.
#     max_labels : int, optional
#         The maximum number of labels per client, by default 10.
#     seed : int, optional
#         Used to set a fixed seed to replicate experiments, by default 42.
#     dataset_name : str
#         Name of the dataset to be used.
#
#     Returns
#     -------
#     Tuple[List[Subset], Dataset]
#         The list of datasets for each client, the test dataset.
#     """
#     trainset, testset = _download_data(dataset_name)
#     prng = np.random.default_rng(seed)
#
#     targets = trainset.targets
#     if isinstance(targets, list):
#         targets = np.array(targets)
#     if isinstance(targets, torch.Tensor):
#         targets = targets.numpy()
#     num_classes = len(set(targets))
#     total_samples = len(trainset)
#     samples_per_client = total_samples // num_clients
#
#     # Generate the number of labels for each client
#     labels_per_client = np.clip(
#         prng.normal(loc=mean_labels, scale=std_labels, size=num_clients).astype(int),
#         min_labels,
#         max_labels,
#     )
#
#     # Create a mapping from each label to the corresponding indices
#     label_indices = {k: np.where(targets == k)[0].tolist() for k in range(num_classes)}
#     for indices in label_indices.values():
#         prng.shuffle(indices)
#
#     # Initialize client indices
#     idx_clients: List[List[int]] = [[] for _ in range(num_clients)]
#
#     for client_id, num_labels in enumerate(labels_per_client):
#         selected_labels = prng.choice(range(num_classes), num_labels, replace=False)
#         client_indices = []
#
#         for label in selected_labels:
#             count_needed = (samples_per_client // num_labels) + (
#                 1 if samples_per_client % num_labels > 0 else 0
#             )
#             count_assigned = 0
#
#             while label_indices[label] and count_assigned < count_needed:
#                 client_indices.append(label_indices[label].pop())
#                 count_assigned += 1
#
#         prng.shuffle(client_indices)
#         idx_clients[client_id] = client_indices[:samples_per_client]
#
#     # Ensure all datapoints are used
#     remaining_indices = [idx for indices in label_indices.values() for idx in indices]
#     prng.shuffle(remaining_indices)
#
#     for client_id in range(num_clients):
#         if len(idx_clients[client_id]) < samples_per_client:
#             required = samples_per_client - len(idx_clients[client_id])
#             idx_clients[client_id].extend(remaining_indices[:required])
#             remaining_indices = remaining_indices[required:]
#
#     trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
#     return trainsets_per_client, testset


def get_idxs_custom(
    num_clients: int,
    datapoints_per_client: List[int],
    labels_per_client: List[int],
    seed=42,
    dataset_name="cifar10",
    download=False,
):
    assert (
        len(datapoints_per_client) == num_clients
    ), "The length of datapoints_per_client must be equal to num_clients."
    assert (
        len(labels_per_client) == num_clients
    ), "The length of labels_per_client must be equal to num_clients."

    trainset, testset = _download_data(dataset_name, download=download)
    prng = np.random.default_rng(seed)

    targets = trainset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    num_classes = len(set(targets))
    total_samples = len(trainset)

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

    return idx_clients, trainset, testset


SEED = 42
DATASET = "cifar10"
NUM_CLIENTS = 50


def replicate_clients(
    datapoints_per_client: List[int],
    labels_per_client: List[int],
):
    prng = np.random.default_rng(SEED)

    label_indices = {
        k: np.where(DATASET.targets == k)[0].tolist() for k in range(DATASET.classes)
    }
    for indices in label_indices.values():
        prng.shuffle(indices)

    idx_clients: List[List[int]] = [[] for _ in range(NUM_CLIENTS)]

    for client_id, (num_samples, num_labels) in enumerate(
        zip(datapoints_per_client, labels_per_client)
    ):
        selected_labels = prng.choice(range(DATASET.classes), num_labels, replace=False)
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

    remaining_indices = [idx for indices in label_indices.values() for idx in indices]
    prng.shuffle(remaining_indices)

    for client_id in range(NUM_CLIENTS):
        if len(idx_clients[client_id]) < datapoints_per_client[client_id]:
            required = datapoints_per_client[client_id] - len(idx_clients[client_id])
            idx_clients[client_id].extend(remaining_indices[:required])
            remaining_indices = remaining_indices[required:]

    return idx_clients


def partition_data_custom(
    num_clients: int,
    datapoints_per_client: List[int],
    labels_per_client: List[int],
    seed=42,
    dataset_name="cifar10",
    download=False,
) -> Tuple[List[Dataset], Dataset]:
    """Partition the data according to given number of datapoints and labels for each client.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data.
    datapoints_per_client : List[int]
        List of the number of datapoints for each client.
    labels_per_client : List[int]
        List of the number of labels for each client.
    seed : int, optional
        Used to set a fixed seed to replicate experiments, by default 42.
    dataset_name : str
        Name of the dataset to be used.

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    idx_clients, trainset, testset = get_idxs_custom(
        num_clients,
        datapoints_per_client,
        labels_per_client,
        seed=seed,
        dataset_name=dataset_name,
        download=download,
    )

    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset


def get_custom_info(
    num_clients: int,
    datapoints_per_client: List[int],
    labels_per_client: List[int],
    seed=42,
    dataset_name="cifar10",
    download=False,
):
    client_idxs, trainset, testset = get_idxs_custom(
        num_clients,
        datapoints_per_client,
        labels_per_client,
        seed=seed,
        dataset_name=dataset_name,
        download=download,
    )
    data_volume = [len(idxs) for idxs in client_idxs]
    data_labels = [
        len(
            set([ex[1] for ex in [trainset[idx] for idx in idxs]])
        )  # | set([ex[1] for ex in testset]))
        for idxs in client_idxs
    ]
    return client_idxs, data_volume, data_labels


def get_data_with_idxs(idxs: list[int], dataset_name="cifar10", download=False):
    trainset, testset = _download_data(dataset_name, download=download)
    return Subset(trainset, idxs), testset


def get_dirichlet_info(
    num_clients, alpha, seed=42, dataset_name="cifar10", download=True
):
    trainset, testset = _download_data(dataset_name, download=download)
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(seed)

    # get the targets
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
        len(
            set([ex[1] for ex in [trainset[idx] for idx in idxs]])
            # | set([ex[1] for ex in testset])
        )
        for idxs in idx_clients
    ]
    return idx_clients, data_volume, data_labels


def partition_data_dirichlet(
    num_clients, alpha, seed=42, dataset_name="cifar10", download=False
) -> Tuple[List[Dataset], Dataset]:
    """Partition according to the Dirichlet distribution.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    alpha: float
        Parameter of the Dirichlet distribution
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name, download=download)
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(seed)

    # get the targets
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

    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset


def _download_data(dataset_name="emnist", download=True) -> Tuple[Dataset, Dataset]:
    """Download the requested dataset. Currently supports cifar10, mnist, and fmnist.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training dataset, the test dataset.
    """
    trainset, testset = None, None

    root = "data"
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    if slurm_job_id is not None:
        root = f"/tmp/{slurm_job_id}/data"
        os.makedirs(root, exist_ok=True)

    if dataset_name == "cifar10":
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
    elif dataset_name == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = MNIST(
            root=root,
            train=True,
            download=download,
            transform=transform_train,
        )
        testset = MNIST(
            root=root,
            train=False,
            download=download,
            transform=transform_test,
        )
    elif dataset_name == "fmnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = FashionMNIST(
            root=root,
            train=True,
            download=download,
            transform=transform_train,
        )
        testset = FashionMNIST(
            root=root,
            train=False,
            download=download,
            transform=transform_test,
        )
    else:
        raise NotImplementedError

    return trainset, testset


# pylint: disable=too-many-locals
def partition_data(
    num_clients, similarity=1.0, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    """Partition the dataset into subsets for each client.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    similarity: float
        Parameter to sample similar data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name)
    trainsets_per_client = []
    # for s% similarity sample iid data per client
    s_fraction = int(similarity * len(trainset))
    prng = np.random.default_rng(seed)
    idxs = prng.choice(len(trainset), s_fraction, replace=False)
    iid_trainset = Subset(trainset, idxs)
    rem_trainset = Subset(trainset, np.setdiff1d(np.arange(len(trainset)), idxs))

    # sample iid data per client from iid_trainset
    all_ids = np.arange(len(iid_trainset))
    splits = np.array_split(all_ids, num_clients)
    for i in range(num_clients):
        c_ids = splits[i]
        d_ids = iid_trainset.indices[c_ids]
        trainsets_per_client.append(Subset(iid_trainset.dataset, d_ids))

    if similarity == 1.0:
        return trainsets_per_client, testset

    tmp_t = rem_trainset.dataset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    targets = tmp_t[rem_trainset.indices]
    num_remaining_classes = len(set(targets))
    remaining_classes = list(set(targets))
    client_classes: List[List] = [[] for _ in range(num_clients)]
    times = [0 for _ in range(num_remaining_classes)]

    for i in range(num_clients):
        client_classes[i] = [remaining_classes[i % num_remaining_classes]]
        times[i % num_remaining_classes] += 1
        j = 1
        while j < 2:
            index = prng.choice(num_remaining_classes)
            class_t = remaining_classes[index]
            if class_t not in client_classes[i]:
                client_classes[i].append(class_t)
                times[index] += 1
                j += 1

    rem_trainsets_per_client: List[List] = [[] for _ in range(num_clients)]

    for i in range(num_remaining_classes):
        class_t = remaining_classes[i]
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if class_t in client_classes[j]:
                act_idx = rem_trainset.indices[idx_k_split[ids]]
                rem_trainsets_per_client[j].append(
                    Subset(rem_trainset.dataset, act_idx)
                )
                ids += 1

    for i in range(num_clients):
        trainsets_per_client[i] = ConcatDataset(
            [trainsets_per_client[i]] + rem_trainsets_per_client[i]
        )

    return trainsets_per_client, testset


def partition_data_label_quantity(
    num_clients, labels_per_client, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    """Partition the data according to the number of labels per client.

    Logic from https://github.com/Xtra-Computing/NIID-Bench/.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    num_labels_per_client: int
        Number of labels per client
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name)
    prng = np.random.default_rng(seed)

    targets = trainset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    num_classes = len(set(targets))
    times = [0 for _ in range(num_classes)]
    contains = []

    for i in range(num_clients):
        current = [i % num_classes]
        times[i % num_classes] += 1
        j = 1
        while j < labels_per_client:
            index = prng.choice(num_classes, 1)[0]
            if index not in current:
                current.append(index)
                times[index] += 1
                j += 1
        contains.append(current)
    idx_clients: List[List] = [[] for _ in range(num_clients)]
    for i in range(num_classes):
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if i in contains[j]:
                idx_clients[j] += idx_k_split[ids].tolist()
                ids += 1
    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset


if __name__ == "__main__":
    partition_data(100, 0.1)
