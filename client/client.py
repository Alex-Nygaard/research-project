import json
import os
import pickle
from typing import Optional
import random

from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from datasets.utils.logging import disable_progress_bar

from config.constants import DEVICE, LOG_DIR
from client.network import Net, train, test, eval_learning
from data.noniid_load_dataset import (
    load_datasets,
    get_dirichlet_idxs,
    get_replicate_idxs,
    load_one_client,
)
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
    def read_one(cls, full_path: str, idx: int):
        with open(full_path, mode="r") as file:
            data = json.load(file)
        key = str(idx)
        return (
            data[key]["idxs"],
            data[key]["batch_size"],
            data[key]["local_epoch"],
            data[key]["data_volume"],
            data[key]["data_labels"],
        )

    @classmethod
    def read_many(cls, full_path) -> tuple[
        list,
        list,
        list,
        list,
        list,
    ]:
        with open(full_path, "r") as file:
            data = json.load(file)

        idxs = [data[key]["idxs"] for key in data.keys()]
        batch_sizes = [data[key]["batch_size"] for key in data.keys()]
        local_epochs = [data[key]["local_epoch"] for key in data.keys()]
        data_volumes = [data[key]["data_volume"] for key in data.keys()]
        data_labels = [data[key]["data_labels"] for key in data.keys()]

        return idxs, batch_sizes, local_epochs, data_volumes, data_labels

    @classmethod
    def write_to_json(cls, data: dict, full_path):
        with open(full_path, mode="w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def read_from_json(cls, full_path):
        with open(full_path, mode="r") as file:
            data = json.load(file)
        return data

    @classmethod
    def generate_simulation_clients(
        cls,
        num_clients: int,
        output_path: str,
        trace_file: str = None,
        batch_size: str = "iid",
        local_epochs: str = "iid",
        data_volume: str = "iid",
        data_labels: str = "iid",
    ) -> str:
        unused_idxs, batches, epochs, datapoints, labels = FlowerClient.read_many(
            trace_file
            if trace_file is not None
            else os.path.join("clients", "testing_clients.json")
        )

        if batch_size == "iid":
            bs = [32] * num_clients
        else:
            bs = batches

        if local_epochs == "iid":
            le = [4] * num_clients
        else:
            le = epochs

        if data_volume == "iid":
            dv = [50_000 // num_clients] * num_clients
        else:
            dv = datapoints

        if data_labels == "iid":
            dl = [10] * num_clients
        else:
            dl = labels

        client_idxs, data_volume, data_labels = get_replicate_idxs(num_clients, dv, dl)

        client_attributes = {}
        for cid, idxs, batch_size, local_epoch, volume, label in zip(
            range(num_clients),
            client_idxs,
            bs,
            le,
            data_volume,
            data_labels,
        ):
            client_attributes[cid] = {
                "cid": cid,
                "idxs": idxs,
                "batch_size": batch_size,
                "local_epoch": local_epoch,
                "data_volume": volume,
                "data_labels": label,
            }

        with open(output_path, mode="w") as file:
            json.dump(client_attributes, file, indent=4)

        return output_path

    @classmethod
    def generate_deployment_clients(
        cls,
        num_clients: int,
        output_paths: list[str],
        seed: int = 42,
    ):
        bs_opts = [16, 32, 64, 128]
        le_opts = [1, 3, 5, 7]

        batch_sizes = [random.choice(bs_opts) for _ in range(num_clients)]
        local_epochs = [random.choice(le_opts) for _ in range(num_clients)]

        client_idxs, data_volumes, data_labels = get_dirichlet_idxs(
            num_clients, seed=seed, download=True
        )

        client_attributes = {}
        for cid, idxs, batch_size, local_epoch, volume, label in zip(
            range(num_clients),
            client_idxs,
            batch_sizes,
            local_epochs,
            data_volumes,
            data_labels,
        ):
            client_attributes[cid] = {
                "cid": cid,
                "idxs": idxs,
                "batch_size": batch_size,
                "local_epoch": local_epoch,
                "data_volume": volume,
                "data_labels": label,
            }

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

        idxs, batch, epoch, datapoint, label = FlowerClient.read_one(
            (
                trace_file
                if trace_file is not None
                else os.path.join(LOG_DIR, "clients.json")
            ),
            cid,
        )

        train_dl, test_dl = load_one_client(idxs, batch, download=False)
        client = FlowerClient(
            cid,
            batch_size=batch,
            local_epochs=epoch,
            train_loader=train_dl,
            test_loader=test_dl,
        )

        return client.to_client()

    return client_fn
