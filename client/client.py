import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from datasets.utils.logging import disable_progress_bar
from torchvision.transforms import ToTensor, Normalize, Compose
from simulation.data import ClientData
from simulation.constants import DEVICE
from client.network import Net, train, test
from data.load_data import get_data_for_client
from client.attribute import Attribute

disable_progress_bar()


class FlowerClient(NumPyClient):

    def __init__(
        self,
        cid: int,
        client_variation: str = "mid",
        data_variation: str = "mid",
    ):
        self.cid = cid
        self.net = Net().to(DEVICE)

        self.batch_size = Attribute("batch_size", client_variation).get()
        self.local_epochs = Attribute("local_epochs", client_variation).get()
        self.num_data_points = Attribute("num_data_points", data_variation).get()
        self.perc_new_data = Attribute("perc_new_data", data_variation).get()
        self.perc_missing_labels = Attribute(
            "perc_missing_labels", data_variation
        ).get()

        self.full_dataset = get_data_for_client(cid)
        self.train_set, self.test_set = self.full_dataset.train_test_split(
            test_size=0.15
        )
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size)

    def fit(self, parameters, config):
        self.net.set_parameters(parameters)
        current_round = config["current_round"]
        train(self.net, self.train_loader, epochs=self.local_epochs)
        return self.net.get_parameters(config={}), len(self.train_loader), {}

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        loss, accuracy = test(self.net, self.test_loader)
        return loss, len(self.test_loader), {"loss": loss, "accuracy": accuracy}


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
    }
    return config
