import logging
import os
from typing import Dict

from datasets import Dataset, load_dataset
from flwr_datasets import FederatedDataset


class DatasetManager:

    def __init__(self, num_clients: int, dataset_name: str, download_dir: str):
        self.num_clients = num_clients
        self.dataset_name = dataset_name
        self.download_dir = download_dir
        self.datasets: Dict[int, Dataset] = {}
        self.server_test_set: Dataset = None
        self.download_data()

    def download_data(self):
        if not os.path.exists(DOWNLOAD_DIR):
            os.makedirs(DOWNLOAD_DIR)

        logging.info(f"Downloading dataset {DATASET} to {DOWNLOAD_DIR}")
        # ds = load_dataset(path=dir, name=DATASET)

        fds = FederatedDataset(dataset=DATASET, partitioners={"train": self.num_clients})
        self.server_test_set = fds.load_full("test")

        for i in range(self.num_clients):
            # TODO
            self.datasets[i] = fds.load_partition(i, "train")

        logging.info("--- DATASET INFO ---")
        logging.info("  Server test size: %d", len(self.server_test_set))
        logging.info("  Client train TOTAL size: %d", sum([len(self.datasets[i]) for i in range(self.num_clients)]))
        logging.info("  Client train size: %d", len(self.datasets[0]))

        # # Now let's split it into train (90%) and validation (10%)
        # client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)
        #
        # trainset = client_dataset_splits["train"]
        # valset = client_dataset_splits["test"]
        #
        # # Now we apply the transform to each batch.
        # trainset = trainset.with_transform(apply_transforms)
        # valset = valset.with_transform(apply_transforms)

