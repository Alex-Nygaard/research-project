import math
import os
import csv

from datasets import Dataset
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from datasets.utils.logging import disable_progress_bar

from config.constants import DEVICE, LOG_DIR, NUM_ROUNDS
from client.network import Net, train, test, eval_learning
from data.load_data import get_data_for_client
from client.attribute import Attribute

disable_progress_bar()


class FlowerClient(NumPyClient):

    columns = [
        "cid",
        "batch_size",
        "local_epochs",
        "num_data_points",
        "perc_new_data",
        "perc_missing_labels",
        "len_train_set",
        "len_test_set",
    ]

    def __init__(
        self,
        cid: int,
        resources: str = "mid",
        concentration: str = "mid",
        variability: str = "mid",
        quality: str = "mid",
        batch_size: int = None,
        local_epochs: int = None,
        num_data_points: int = None,
        perc_new_data: float = None,
        perc_missing_labels: float = None,
        len_train_set: int = None,
        len_test_set: int = None,
    ):
        self.cid = cid
        self.net = Net().to(DEVICE)

        self.batch_size = batch_size or Attribute("batch_size", resources).generate()
        self.local_epochs = (
            local_epochs or Attribute("local_epochs", resources).generate()
        )
        self.num_data_points = (
            num_data_points or Attribute("num_data_points", concentration).generate()
        )
        self.num_clients = Attribute("num_clients", concentration).generate()
        self.perc_new_data = (
            perc_new_data or Attribute("perc_new_data", variability).generate()
        )
        self.perc_missing_labels = (
            perc_missing_labels or Attribute("perc_missing_labels", quality).generate()
        )

        try:
            self.dataset_dict = get_data_for_client(
                cid,
                self.num_data_points,
            )
            # self.len_train_set = len(self.train_set)
            # self.len_test_set = len(self.test_set)
        except Exception as e:
            print(f"XXXXXX Error: {e}")
            # self.train_set: Dataset = None
            # self.test_set: Dataset = None
            self.len_train_set = len_train_set if len_train_set is not None else 0
            self.len_test_set = len_test_set if len_test_set is not None else 0

    def fit(self, parameters, config):
        self.net.set_parameters(parameters)
        self.net.train()
        current_round = config["current_round"]

        idx = math.floor(
            (self.num_data_points * 0.85) / (1 + (current_round - 1) * self.perc_new_data)
        )

        print(
            f"idx={idx},current_round={current_round},num_data_points={self.num_data_points},perc_new_data={self.perc_new_data}"
        )

        data = self.dataset_dict["train"].select(range(idx))

        train_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        train(self.net, train_loader, epochs=self.local_epochs)
        return self.net.get_parameters(config={}), len(train_loader), {}

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        self.net.eval()
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size)
        loss, accuracy = test(self.net, test_loader)

        acc, rec, prec, f1 = eval_learning(self.net, test_loader)
        output_dict = {
            "accuracy": accuracy,
            "acc": acc,
            "rec": rec,
            "prec": prec,
            "f1": f1,
            "loss": loss,
        }

        return loss, len(test_loader), output_dict

    def set_parameters(self, parameters):
        self.net.set_parameters(parameters)

    def get_parameters(self, config):
        return self.net.get_parameters(config=config)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @staticmethod
    def of(
        cid,
        batch_size,
        local_epochs,
        num_data_points,
        perc_new_data,
        perc_missing_labels,
        len_train_set,
        len_test_set,
    ):
        return FlowerClient(
            cid=cid,
            batch_size=batch_size,
            local_epochs=local_epochs,
            num_data_points=num_data_points,
            perc_new_data=perc_new_data,
            perc_missing_labels=perc_missing_labels,
            len_train_set=len_train_set,
            len_test_set=len_test_set,
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


def fit_config(server_round: int):
    config = {
        "current_round": server_round,
    }
    return config


def get_client_fn(
    resources: str = "mid",
    concentration: str = "mid",
    variability: str = "mid",
    quality: str = "mid",
    deployment_id: int = None,
):
    def client_fn(simulation_id: str):
        cid = int(deployment_id) if deployment_id is not None else int(simulation_id)

        clients = FlowerClient.read_from_csv(LOG_DIR, "clients.csv")
        for client in clients:
            if client.cid == cid:
                return client.to_client()

        client = FlowerClient(
            cid,
            resources=resources,
            concentration=concentration,
            variability=variability,
            quality=quality,
        )
        client.write_to_csv(LOG_DIR, "clients.csv")
        return client.to_client()

    return client_fn
