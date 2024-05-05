import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import logging

import flwr as fl
from flwr_datasets import FederatedDataset

import argparse
import warnings
from collections import OrderedDict

from datasets.utils.logging import disable_progress_bar

from flwr.client import NumPyClient, ClientApp
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

logger = logging.getLogger()

DATASET = "cifar10"
DOWNLOAD_DIR = os.getcwd() + "/data"
NUM_CLIENTS = 5

torch_device = "cpu"
if torch.cuda.is_available():
    torch_device = "cuda"
elif torch.backends.mps.is_available():
    torch_device = "mps"
logger.info("TORCH Using %s", torch_device)
DEVICE = torch.device(torch_device)


logger.info("Downloading dataset %s to %s", DATASET, DOWNLOAD_DIR)
fds = FederatedDataset(dataset=DATASET, partitioners={"train": 10})
logger.info("Federated dataset %s loaded", DATASET)
centralized_testset = fds.load_split("test")
logger.info("Server test size: %d", len(centralized_testset))


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for batch in tqdm(trainloader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def load_data(partition_id):
    """Load partition CIFAR10 data."""
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


# Define Flower client
class FlowerClient(NumPyClient):

    def __init__(self, cid: int):
        self.trainloader, self.testloader = load_data(cid)
        self.net = Net().to(DEVICE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = 1 if "epochs" not in config else config["epochs"]
        train(self.net, self.trainloader, epochs=epochs)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 3,  # Number of local epochs done by clients
        "batch_size": 32,  # Batch size to use by clients during fit()
    }
    return config

def client_fn(cid: str) -> fl.client.Client:
    return FlowerClient(int(cid)).to_client()


def evaluate_strategy(
    server_round, parameters, config
):
    """Use the entire CIFAR-10 test set for evaluation."""

    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Apply transform to dataset
    testset = centralized_testset.with_transform(apply_transforms)

    # Disable tqdm for dataset preprocessing
    disable_progress_bar()

    client = FlowerClient(0)
    client.set_parameters(parameters)
    model = client.net

    testloader = DataLoader(testset, batch_size=32)
    loss, accuracy = test(model, testloader)

    return loss, {"accuracy": accuracy, "loss": loss}


strategy = fl.server.strategy.FedAvg(
    on_fit_config_fn=fit_config,  # Configuration for fit
    evaluate_fn=evaluate_strategy,
)

def main():
    logger.info("Starting simulation")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
    )


if __name__ == "__main__":
    main()
