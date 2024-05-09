from collections import OrderedDict
from typing import Dict, Tuple, List

import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import Scalar

from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset

from simulation.client import get_client_fn
from simulation.strategy import FedCustom, evaluate_fn, evaluate_metrics_aggregation_fn

"""
def start_simulation(
    *,
    client_fn: ClientFn,
    num_clients: Optional[int] = None,
    clients_ids: Optional[List[str]] = None,
    client_resources: Optional[Dict[str, float]] = None,
    server: Optional[Server] = None,
    config: Optional[ServerConfig] = None,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    ray_init_args: Optional[Dict[str, Any]] = None,
    keep_initialised: Optional[bool] = False,
    actor_type: Type[VirtualClientEngineActor] = ClientAppActor,
    actor_kwargs: Optional[Dict[str, Any]] = None,
    actor_scheduling: Union[str, NodeAffinitySchedulingStrategy] = "DEFAULT",
) -> History:
"""


def run_simulation():
    print("Running simulation")

    fl.simulation.start_simulation(
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=2),
        client_fn=get_client_fn,
        strategy=FedCustom(
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        ),
    )
