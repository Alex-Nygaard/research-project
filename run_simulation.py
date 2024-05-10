import flwr as fl
import argparse
from strategy.strategy import get_strategy
from config.constants import NUM_ROUNDS
from logger.logger import get_logger

from client.client import get_client_fn, fit_config
from strategy.strategy import (
    FedCustom,
    evaluate_fn,
    evaluate_metrics_aggregation_fn,
)
from config.constants import NUM_ROUNDS, NUM_CLIENTS
from logger.logger import get_logger

logger = get_logger("simulation.run")

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

    logger.info(
        "Running simulation with client variation: %s and data variation: %s",
        client_variation,
        data_variation,
    )

    fl.simulation.start_simulation(
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        client_fn=get_client_fn(client_variation, data_variation),
        strategy=FedCustom(
            evaluate_fn=evaluate_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            on_fit_config_fn=fit_config,
        ),
        # TODO ray_init_args=ray_init_args,
    )
