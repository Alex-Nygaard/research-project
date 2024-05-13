import os
import flwr as fl
import argparse
from strategy.strategy import get_strategy

from client.client import get_client_fn
from config.constants import NUM_ROUNDS, NUM_CLIENTS, LOG_DIR
from logger.logger import get_logger
from data.save_results import save_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running a simulation.")
    parser.add_argument(
        "--data_variation",
        type=str,
        choices=["low", "mid", "high"],
        help="Partition ID",
    )
    parser.add_argument(
        "--client_variation",
        type=str,
        choices=["low", "mid", "high"],
        help="Partition ID",
    )

    args = parser.parse_args()
    data_variation = args.data_variation
    client_variation = args.client_variation

    logger = get_logger("simulation.run")
    fl.common.logger.configure(
        identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
    )

    logger.info(
        "Running simulation with client variation: %s and data variation: %s",
        client_variation,
        data_variation,
    )

    history = fl.simulation.start_simulation(
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        client_fn=get_client_fn(client_variation, data_variation),
        strategy=get_strategy(),
        ray_init_args={
            "num_cpus": os.environ.get("SLURM_CPUS_PER_TASK", 8),
            "num_gpus": os.environ.get("SLURM_GPUS", 0),
        },
    )

    save_history(history)
