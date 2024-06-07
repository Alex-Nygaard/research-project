import json
import os
from typing import Optional
import random

from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from datasets.utils.logging import disable_progress_bar

from config.constants import DEVICE, LOG_DIR
from client.network import Net, train, test, eval_learning
from data.noniid_load_dataset import load_datasets
from logger.logger import get_logger

disable_progress_bar()

log = get_logger("client.client")


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

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.attributes = self.Attributes(
            cid=cid,
            batch_size=batch_size,
            local_epochs=local_epochs,
            data_volume=data_volume,
            data_labels=data_labels,
        )

    def fit(self, parameters, config):
        self.net.set_parameters(parameters)
        self.net.train()
        current_round = config["current_round"]

        train(self.net, self.train_loader, epochs=self.attributes.local_epochs)
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
    def write_to_json(cls, data: dict, full_path):
        with open(full_path, mode="w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def generate_deployment_clients(
        cls,
        num: int,
        output_paths: list[str],
    ):
        bs_opts = [16, 32, 64, 128]
        le_opts = [1, 3, 5, 7]

        batch_sizes = [random.choice(bs_opts) for _ in range(num)]
        local_epochs = [random.choice(le_opts) for _ in range(num)]

        train_datasets, test_datasets, valid_set = load_datasets(
            {
                "partitioning": "dirichlet",
                "alpha": 0.5,
                "batch_sizes": batch_sizes,
                "name": "cifar10",
            },
            num_clients=num,
        )

        client_attributes = {}
        for cid, train_dl, test_dl, batch_size, local_epoch in zip(
            range(num), train_datasets, test_datasets, batch_sizes, local_epochs
        ):
            client = FlowerClient(
                cid=cid,
                batch_size=batch_size,
                local_epochs=local_epoch,
                data_volume=len(train_dl.dataset) + len(test_dl.dataset),
                data_labels=len(
                    set([ex[1] for ex in train_dl.dataset])
                    | set([ex[1] for ex in train_dl.dataset])
                ),
            )
            client_attributes[cid] = client.attributes.__dict__

        for path in output_paths:
            log.info(f"Writing client attributes to {path}...")
            cls.write_to_json(client_attributes, path)


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

        assert (
            len(batches) == len(epochs) == len(datapoints) == len(labels) == num_clients
        ), "Number of clients does not match trace file."

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
