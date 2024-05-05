print("hello")

import os
import sys

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import logging

import flwr as fl
from flwr_datasets import FederatedDataset
from datasets import Dataset

from experiment.client import fit_config, get_client_fn
from experiment.strategy import FedCustom
from experiment.common import DatasetManager
from datasets.utils.logging import disable_progress_bar


DATASET = "cifar100"
NUM_CLIENTS = 5
NUM_ROUNDS = 10
DOWNLOAD_DIR = "data"

logger = logging.getLogger(__file__)
logger.info(f"Running simulation with {NUM_CLIENTS} clients for {NUM_ROUNDS} rounds.")

# Download MNIST dataset and partition it
logger.info("Downloading dataset %s to %s", DATASET, DOWNLOAD_DIR)
mnist_fds = FederatedDataset(dataset=DATASET, partitioners={"train": NUM_CLIENTS})
logger.info("Federated dataset %s loaded", DATASET)
centralized_testset = mnist_fds.load_split("test")
logger.info("Server test size: %d", len(centralized_testset))


def main():
    logger.info("Starting simulation")

    fl.simulation.start_simulation(
        client_fn=get_client_fn(mnist_fds),
        num_clients=NUM_CLIENTS,
        # client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=FedCustom(on_fit_config_fn=fit_config),
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
        ray_init_args={'num_cpus': 8, 'num_gpus': 0}
    )


if __name__ == "__main__":
    main()
