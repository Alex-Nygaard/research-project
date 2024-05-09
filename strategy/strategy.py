from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from flwr.common import Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
import flwr as fl
from datasets.utils.logging import disable_progress_bar

from client.network import Net, test
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

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses and accuracy using weighted average."""

        if not results:
            return None, {}

        # Calculate weighted average for loss using the provided function
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        accuracies = [
            evaluate_res.metrics["accuracy"] * evaluate_res.num_examples
            for _, evaluate_res in results
        ]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        accuracy_aggregated = (
            sum(accuracies) / sum(examples) if sum(examples) != 0 else 0
        )

        metrics_aggregated = {"loss": loss_aggregated, "accuracy": accuracy_aggregated}

        return loss_aggregated, metrics_aggregated


def evaluate_fn(
    server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
):
    device = torch.device(DEVICE)

    model = Net()
    model.to(device)
    model.set_parameters(parameters)

    testloader = DataLoader(centralized_test_set, batch_size=32)
    loss, accuracy = test(model, testloader)

    logger.info(
        "[CENTRAL EVAL]: Round %s, Loss: %s, Accuracy: %s", server_round, loss, accuracy
    )

    return loss, {"accuracy": accuracy, "loss": loss}


def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, Scalar]]]):
    aggregated_metrics = {}

    if "accuracy" in metrics[0][1]:
        accuracies = [
            metric["accuracy"] * num_examples for num_examples, metric in metrics
        ]
        examples = [num_examples for num_examples, evaluate_res in metrics]
        accuracy_aggregated = (
            sum(accuracies) / sum(examples) if sum(examples) != 0 else 0
        )
        aggregated_metrics["accuracy"] = accuracy_aggregated

    if "loss" in metrics[0][1]:
        losses = [metric["loss"] * num_examples for num_examples, metric in metrics]
        examples = [num_examples for num_examples, evaluate_res in metrics]
        loss_aggregated = sum(losses) / sum(examples) if sum(examples) != 0 else 0
        aggregated_metrics["loss"] = loss_aggregated

    return aggregated_metrics


def get_strategy():
    fraction_fit = 1.0
    fraction_evaluate = 1.0
    logger.info(
        "Using custom strategy. (fraction_fit=%s, fraction_evaluate=%s)",
        fraction_fit,
        fraction_evaluate,
    )
    return FedCustom(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config,
    )
