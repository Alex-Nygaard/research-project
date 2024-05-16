import os
import flwr as fl
import argparse

from client.attribute import Attribute
from strategy.strategy import get_strategy
from client.client import get_client_fn
from config.constants import NUM_ROUNDS, LOG_DIR
from logger.logger import get_logger
from data.save_results import save_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running a simulation.")
    parser.add_argument(
        "--data_variation",
        type=str,
        choices=["low", "mid", "high"],
        help="Data variation.",
    )
    parser.add_argument(
        "--client_variation",
        type=str,
        choices=["low", "mid", "high"],
        help="Client variation.",
    )

    args = parser.parse_args()
    data_variation = args.data_variation
    client_variation = args.client_variation

    logger = get_logger("simulation.run")
    fl.common.logger.configure(
        identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
    )

    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    num_gpus = int(os.environ.get("SLURM_GPUS_PER_TASK", 0))

    logger.info(
        "RUNNING SIMULATION with client_var=%s,data_var=%s,num_cpus=%s,num_gpus=%s",
        client_variation,
        data_variation,
        num_cpus,
        num_gpus,
    )

    history = fl.simulation.start_simulation(
        num_clients=Attribute("num_clients", client_variation).generate(),
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        client_fn=get_client_fn(client_variation, data_variation),
        strategy=get_strategy(),
        ray_init_args={
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
        },
    )

    save_history(history)
