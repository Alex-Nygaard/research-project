from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from flwr.common import Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
import flwr as fl
from datasets.utils.logging import disable_progress_bar

from client.network import Net, test, eval_learning
from client.client import fit_config
from config.constants import DEVICE
from data.load_data import centralized_test_set
from logger.logger import get_logger

disable_progress_bar()

logger = get_logger("strategy.strategy")


class FedCustom(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return "FedCustom"


def evaluate_fn(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
):
    device = torch.device(DEVICE)

    model = Net()
    model.to(device)
    model.set_parameters(parameters)
    model.eval()

    test_loader = DataLoader(centralized_test_set, batch_size=32)
    loss, accuracy = test(model, test_loader)

    acc, rec, prec, f1 = eval_learning(model, test_loader)
    output_dict = {
        "accuracy": accuracy,
        "acc": acc,
        "rec": rec,
        "prec": prec,
        "f1": f1,
    }

    return loss, output_dict


def average_metrics(metrics):
    accuracies_tf = np.mean([metric["accuracy"] for _, metric in metrics])
    accuracies = np.mean([metric["acc"] for _, metric in metrics])
    recalls = np.mean([metric["rec"] for _, metric in metrics])
    precisions = np.mean([metric["prec"] for _, metric in metrics])
    f1s = np.mean([metric["f1"] for _, metric in metrics])

    return {
        "accuracy": accuracies_tf,
        "acc": accuracies,
        "rec": recalls,
        "prec": precisions,
        "f1": f1s,
    }


def get_strategy():
    fraction_fit = 0.5
    fraction_evaluate = 0.5
    logger.info(
        "Using custom strategy. (fraction_fit=%s, fraction_evaluate=%s)",
        fraction_fit,
        fraction_evaluate,
    )
    return FedCustom(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=average_metrics,
        on_fit_config_fn=fit_config,
    )
