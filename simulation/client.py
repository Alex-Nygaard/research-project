import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from datasets.utils.logging import disable_progress_bar
from torchvision.transforms import ToTensor, Normalize, Compose
from simulation.data import ClientData
from simulation.constants import DEVICE

disable_progress_bar()


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

    def set_parameters(self, parameters):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for batch in trainloader:
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
        for batch in testloader:
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


# def load_data(partition_id, batch_size=32):
#     partition = federated_dataset.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#
#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(
#         partition_train_test["train"], batch_size=batch_size, shuffle=True
#     )
#     testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
#     return trainloader, testloader


class SimulationClient(NumPyClient):

    def __init__(self, cid: int, trainloader, testloader):
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = Net().to(DEVICE)

    def fit(self, parameters, config):
        self.net.set_parameters(parameters)
        epochs = 2 if "epochs" not in config else config["epochs"]
        train(self.net, self.trainloader, epochs=epochs)
        return self.net.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"loss": loss, "accuracy": accuracy}


def get_client_fn(cid: str):
    return SimulationClient(int(cid)).to_client()


class Client:
    def __init__(
        self,
        cid: int,
        num_data_points: int,
        label_count: int,
        local_epochs: int,
        batch_size: int,
        dropout_chance: float,
    ):
        self.cid = cid
        self.num_data_points = num_data_points
        self.label_count = label_count
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.dropout_chance = dropout_chance

        self.dataset = ClientData.shard_data(client_id=cid)

        self.simulation_client = SimulationClient(cid).to_client()

    def get_simulation_client(self):
        return self.simulation_client

    def __repr__(self):
        return f"Client(id={self.cid}, num_data_points={self.num_data_points}, label_count={self.label_count}, local_epochs={self.local_epochs}, batch_size={self.batch_size}, dropout_chance={self.dropout_chance})"

    @staticmethod
    def generate_ids(n):
        return list(range(n))
