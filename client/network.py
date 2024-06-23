import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from datasets.utils.logging import disable_progress_bar
from collections import OrderedDict
from config.constants import DEVICE

disable_progress_bar()


class Net(nn.Module):
    """
    Simple CNN model for CIFAR-10 dataset.

    Inspired by: https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html
    """

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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for _ in range(epochs):
        for batch in trainloader:
            assert len(batch) == 2, f"Batch should have 2 elements, got {len(batch)}"
            images = batch[0]
            labels = batch[1]
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            assert len(batch) == 2, f"Batch should have 2 elements, got {len(batch)}"
            images = get_data(batch).to(DEVICE)
            labels = get_label(batch).to(DEVICE)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item() * labels.size(0)
    accuracy = correct / len(testloader.dataset)
    loss /= len(testloader.dataset)
    return loss, accuracy


def eval_learning(net: Net, testloader: DataLoader):
    with torch.no_grad():
        images = torch.cat([get_data(batch).to(DEVICE) for batch in testloader], dim=0)
        labels = torch.cat([get_label(batch).to(DEVICE) for batch in testloader], dim=0)

        outputs = net(images)

        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        y_actual = labels.cpu().numpy()

    acc = accuracy_score(y_actual, y_pred)
    rec = recall_score(
        y_actual, y_pred, average="micro"
    )  # average argument required for multi-class
    prec = precision_score(y_actual, y_pred, average="micro")
    f1 = f1_score(y_actual, y_pred, average="micro")
    return acc, rec, prec, f1


def get_data(batch):
    return batch["img"] if "img" in batch else batch[0]


def get_label(batch):
    return batch["label"] if "label" in batch else batch[1]
