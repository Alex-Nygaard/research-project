import json
import math
import os
import csv
from typing import List, Tuple, Optional

from datasets import Dataset
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from datasets.utils.logging import disable_progress_bar

from config.constants import DEVICE, LOG_DIR
from client.network import Net, train, test, eval_learning
from data.load_data import get_data_for_client, apply_transforms
from data.noniid_load_dataset import load_datasets
from client.attribute import Attribute

disable_progress_bar()


class FlowerClient(NumPyClient):

    class Attributes:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    batch_size_options = [16, 32, 64, 128]
    local_epoch_options = [1, 3, 5, 7]

    def __init__(
        self,
        cid: int,
        batch_size: Optional[int] = None,
        local_epochs: Optional[int] = None,
        data_volume: Optional[int] = None,
        data_labels: Optional[int] = None,
        train_loader: DataLoader = None,
        test_loader: DataLoader = None,
    ):
        self.cid = cid
        self.net = Net().to(DEVICE)

        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.data_volume = len(train_loader.dataset) + len(test_loader.dataset)

    def fit(self, parameters, config):
        self.net.set_parameters(parameters)
        self.net.train()
        current_round = config["current_round"]

        train(self.net, self.train_loader, epochs=self.local_epochs)
        return self.net.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        self.net.eval()
        loss, accuracy = test(self.net, self.test_loader)

        acc, rec, prec, f1 = eval_learning(self.net, self.test_loader)
        output_dict = {
            "accuracy": accuracy,
            "acc": acc,
            "rec": rec,
            "prec": prec,
            "f1": f1,
            "loss": loss,
        }

        return loss, len(self.test_loader.dataset), output_dict

    def set_parameters(self, parameters):
        self.net.set_parameters(parameters)

    def get_parameters(self, config):
        return self.net.get_parameters(config=config)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @classmethod
    def of(
        cls,
        cid,
        batch_size_dist: str = "iid",
        local_epochs_dist: str = "iid",
        data_volume_dist: str = "iid",
        data_labels_dist: str = "iid",
    ):
        batch_size = (
            32
            if batch_size_dist == "iid"
            else cls.batch_size_options[cid % len(cls.batch_size_options)]
        )
        local_epochs = (
            5
            if local_epochs_dist == "iid"
            else cls.local_epoch_options[cid % len(cls.local_epoch_options)]
        )

        return FlowerClient(
            cid=cid,
            batch_size=batch_size,
            local_epochs=local_epochs,
        )

    def write_to_csv(self, path, filename):
        data = [str(self[col]) for col in FlowerClient.columns]
        with open(os.path.join(path, filename), "a") as file:
            file.write(",".join(data) + "\n")
            file.flush()

    @staticmethod
    def write_many(clients, path, filename):
        for client in clients:
            client.write_to_csv(path, filename)

    @classmethod
    def read_from_csv(cls, path, filename):
        full_path = os.path.join(path, filename)

        clients = []

        with open(full_path, mode="r", newline="") as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # skip header
            for row in csv_reader:
                kwargs = {
                    col: Attribute.get(col, value)
                    for col, value in zip(cls.columns, row)
                }
                clients.append(cls.of(**kwargs))

        return clients

    @classmethod
    def read_attributes_from_json(cls, full_path) -> tuple[
        list,
        list,
        list,
        list,
    ]:
        with open(full_path, "r") as file:
            data = json.load(file)

        batch_sizes = [data[key]["batch_size"] for key in data.keys()]
        local_epochs = [data[key]["local_epochs"] for key in data.keys()]
        data_volume = [data[key]["data_volume"] for key in data.keys()]
        data_labels = [data[key]["data_labels"] for key in data.keys()]

        return batch_sizes, local_epochs, data_volume, data_labels

    @classmethod
    def write_attributes_to_json(cls, data: dict, path, filename):
        full_path = os.path.join(path, filename)
        with open(full_path, mode="w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def generate_deployment_clients(
        cls,
        num: int,
        path: str,
        filename: str,
    ):
        bs_opts = [16, 32, 64, 128]
        le_opts = [1, 3, 5, 7]

        batch_sizes = [bs_opts[i % len(bs_opts)] for i in range(num)]
        local_epochs = [le_opts[i % len(le_opts)] for i in range(num)]

        train_datasets, test_datasets, valid_set = load_datasets(
            {
                "partitioning": "dirichlet",
                "alpha": 0.5,
                "batch_sizes": batch_sizes,
                "name": "cifar10",
            },
            num_clients=num,
        )

        clients = {}
        for cid, train_dl, test_dl, batch_size, local_epoch in zip(
            range(num), train_datasets, test_datasets, batch_sizes, local_epochs
        ):
            client = {
                "cid": cid,
                "batch_size": batch_size,
                "local_epochs": local_epoch,
                "data_volume": len(train_dl.dataset) + len(test_dl.dataset),
                "data_labels": len(
                    set([ex[1] for ex in train_dl.dataset])
                    | set([ex[1] for ex in train_dl.dataset])
                ),
            }
            clients[cid] = client

        cls.write_attributes_to_json(clients, path, filename)


def fit_config(server_round: int):
    config = {
        "current_round": server_round,
    }
    return config


def get_client_fn(
    num_clients: int,
    option: str = "simulation",
    trace_file: str = None,
    batch_sizes: str = "iid",
    local_epochs: str = "iid",
    data_volume: str = "iid",
    data_labels: str = "iid",
    deployment_id: int = None,
):
    def client_fn(simulation_id: str):
        cid = int(deployment_id) if deployment_id is not None else int(simulation_id)

        batches, epochs, datapoints, labels = FlowerClient.read_attributes_from_json(
            trace_file
            if trace_file is not None
            else os.path.join(LOG_DIR, "clients.json")
        )
        if option == "simulation":
            if batch_sizes == "iid":
                batches = [32] * num_clients
            if local_epochs == "iid":
                epochs = [4] * num_clients
            if data_volume == "iid":
                datapoints = [50_000 // num_clients] * num_clients
            if data_labels == "iid":
                labels = [10] * num_clients

            train_datasets, test_datasets, valid_set = load_datasets(
                {
                    "partitioning": "replicate",
                    "alpha": 0.5,
                    "batch_sizes": batches,
                    "datapoints_per_client": datapoints,
                    "labels_per_client": labels,
                    "name": "cifar10",
                },
                num_clients=num_clients,
            )
        else:
            train_datasets, test_datasets, valid_set = load_datasets(
                {
                    "partitioning": "dirichlet",
                    "alpha": 0.5,
                    "batch_sizes": batches,
                    "name": "cifar10",
                },
                num_clients=num_clients,
            )

        client = FlowerClient(
            cid,
            batch_size=batches[cid],
            local_epochs=epochs[cid],
            train_loader=train_datasets[cid],
            test_loader=test_datasets[cid],
        )

        return client.to_client()

    return client_fn
