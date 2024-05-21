import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from datasets.utils.logging import disable_progress_bar
from collections import OrderedDict
from config.constants import DEVICE

disable_progress_bar()


#
# class Block(nn.Module):
#     """expand + depthwise + pointwise"""
#
#     def __init__(self, in_planes, out_planes, expansion, stride):
#         super(Block, self).__init__()
#         self.stride = stride
#
#         planes = expansion * in_planes
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes,
#             planes,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             groups=planes,
#             bias=False,
#         )
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(
#             planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False
#         )
#         self.bn3 = nn.BatchNorm2d(out_planes)
#
#         self.shortcut = nn.Sequential()
#         if stride == 1 and in_planes != out_planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_planes,
#                     out_planes,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(out_planes),
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out = out + self.shortcut(x) if self.stride == 1 else out
#         return out
#
#
# class Net(nn.Module):
#     # (expansion, out_planes, num_blocks, stride)
#     cfg = [
#         (1, 16, 1, 1),
#         (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
#         (6, 32, 3, 2),
#         (6, 64, 4, 2),
#         (6, 96, 3, 1),
#         (6, 160, 3, 2),
#         (6, 320, 1, 1),
#     ]
#
#     def __init__(self, num_classes=10):
#         super(Net, self).__init__()
#         # NOTE: change conv1 stride 2 -> 1 for CIFAR10
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.layers = self._make_layers(in_planes=32)
#         self.conv2 = nn.Conv2d(
#             320, 1280, kernel_size=1, stride=1, padding=0, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(1280)
#         self.linear = nn.Linear(1280, num_classes)
#
#     def _make_layers(self, in_planes):
#         layers = []
#         for expansion, out_planes, num_blocks, stride in self.cfg:
#             strides = [stride] + [1] * (num_blocks - 1)
#             for stride in strides:
#                 layers.append(Block(in_planes, out_planes, expansion, stride))
#                 in_planes = out_planes
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layers(out)
#         out = F.relu(self.bn2(self.conv2(out)))
#         # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
#
#     def set_parameters(self, parameters):
#         params_dict = zip(self.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         self.load_state_dict(state_dict, strict=True)
#
#     def get_parameters(self, config):
#         return [val.cpu().numpy() for _, val in self.state_dict().items()]


class Net(nn.Module):
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
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item() * labels.size(0)
    accuracy = correct / len(testloader.dataset)
    loss /= len(testloader.dataset)
    return loss, accuracy


def eval_learning(net: Net, testloader: DataLoader):
    with torch.no_grad():
        images = torch.cat([batch["img"].to(DEVICE) for batch in testloader], dim=0)
        labels = torch.cat([batch["label"].to(DEVICE) for batch in testloader], dim=0)

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
