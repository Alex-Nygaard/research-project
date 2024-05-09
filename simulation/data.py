import datasets
from flwr_datasets import FederatedDataset
from datasets import Dataset
from torch.utils.data import DataLoader
import numpy as np
from simulation.constants import NUM_CLIENTS


DATASET = "cifar10"
SAVEPATH = "data"

try:
    dataset = datasets.load_from_disk(SAVEPATH, keep_in_memory=True)
except FileNotFoundError:
    print("Dataset not found, downloading...")
    dataset = datasets.load_dataset(path=DATASET, keep_in_memory=True)
    dataset.save_to_disk(SAVEPATH)

train_set = dataset["train"]
print("[DATA] Client training examples: ", len(train_set))
test_set = dataset["test"]
print("[DATA] Centralized testset examples: ", len(test_set))

train_set.shuffle(seed=1010)
test_set.shuffle(seed=1010)


class ClientData:
    def __init__(self, dataset_shard: Dataset, batch_size):
        split_dict = dataset_shard.train_test_split(test_size=0.2, seed=1010)
        self.train_loader = DataLoader(
            split_dict["train"], batch_size=batch_size, shuffle=True
        )
        self.test_loader = DataLoader(split_dict["test"], batch_size=batch_size)

        self.num_train_examples = len(self.train_loader)
        self.num_test_examples = len(self.test_loader)
        self.batch_size = batch_size

    @staticmethod
    def shard_data(client_id: int) -> Dataset:
        shard = train_set.shard(num_shards=NUM_CLIENTS, index=client_id)
        return shard


class ServerData:
    def __init__(self, batch_size=64):
        self.loader = DataLoader(test_set, batch_size=batch_size)
        self.num_examples = len(self.loader)
        self.batch_size = batch_size

    def get_data(self):
        for data in self.loader:
            yield data


if __name__ == "__main__":
    print("simulation/data.py")
