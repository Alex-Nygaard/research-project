from torch.utils.data import DataLoader
from flwr.client import NumPyClient
from datasets.utils.logging import disable_progress_bar

from config.constants import DEVICE
from client.network import Net, train, test, eval_learning
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
        split = self.full_dataset.train_test_split(test_size=0.15)
        self.train_set, self.test_set = split["train"], split["test"]

    def fit(self, parameters, config):
        self.net.set_parameters(parameters)
        self.net.train()
        current_round = config["current_round"]
        train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True
        )
        train(self.net, train_loader, epochs=self.local_epochs)
        return self.net.get_parameters(config={}), len(train_loader), {}

    def evaluate(self, parameters, config):
        self.net.set_parameters(parameters)
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size)
        loss, accuracy = test(self.net, test_loader)

        acc, rec, prec, f1 = eval_learning(self.net, test_loader)
        output_dict = {
            "accuracy": accuracy,
            "acc": acc,
            "rec": rec,
            "prec": prec,
            "f1": f1,
        }

        return loss, len(test_loader), output_dict

    def set_parameters(self, parameters):
        self.net.set_parameters(parameters)

    def get_parameters(self, config):
        return self.net.get_parameters(config=config)


def fit_config(server_round: int):
    config = {
        "current_round": server_round,
    }
    return config


def get_client_fn(
    client_variation: str = "mid",
    data_variation: str = "mid",
):
    def client_fn(cid: str):
        return FlowerClient(int(cid), client_variation, data_variation).to_client()

    return client_fn
