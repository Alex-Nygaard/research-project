import os

import datasets
from flwr_datasets import FederatedDataset
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np
from simulation.constants import NUM_CLIENTS


DATASET = "cifar10"
SAVEPATH = "data/storage"

try:
    dataset = datasets.load_from_disk(SAVEPATH, keep_in_memory=True)
except FileNotFoundError:
    print("Dataset not found, downloading...")
    dataset = datasets.load_dataset(path=DATASET, keep_in_memory=True)
    if not os.path.exists(SAVEPATH):
        os.makedirs(SAVEPATH)
    dataset.save_to_disk(SAVEPATH)

train_set = dataset["train"]
print("[DATA] Client training examples: ", len(train_set))
centralized_test_set = dataset["test"]
print("[DATA] Centralized testset examples: ", len(centralized_test_set))

train_set.shuffle(seed=1010)
centralized_test_set.shuffle(seed=1010)


def get_data_for_client(cid: int):
    return train_set.shard(num_shards=NUM_CLIENTS, index=cid)
