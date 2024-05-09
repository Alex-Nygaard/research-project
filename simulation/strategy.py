from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from flwr.common import Scalar, EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import flwr as fl
import logging
from prometheus_client import Gauge
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar

from simulation.client import Net, train, test, apply_transforms
from simulation.client import DEVICE


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
    model.set_parameters(parameters)
    model.to(device)

    # Apply transform to dataset
    testset = centralized_testset.with_transform(apply_transforms)

    # Disable tqdm for dataset preprocessing
    disable_progress_bar()

    testloader = DataLoader(testset, batch_size=32)
    loss, accuracy = test(model, testloader)

    return loss, {"accuracy": accuracy, "loss": loss}


def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, Scalar]]]):
    accuracies = [metric["accuracy"] * num_examples for num_examples, metric in metrics]

    examples = [num_examples for num_examples, evaluate_res in metrics]
    accuracy_aggregated = sum(accuracies) / sum(examples) if sum(examples) != 0 else 0

    metrics_aggregated = {"accuracy": accuracy_aggregated}

    return metrics_aggregated
