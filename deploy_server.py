import os
import flwr as fl
from strategy.strategy import get_strategy
from config.constants import NUM_ROUNDS, LOG_DIR
from logger.logger import get_logger
from data.save_results import save_history


if __name__ == "__main__":
    logger = get_logger("server.deploy")
    fl.common.logger.configure(
        identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
    )

    logger.info("Starting server.")
    logger.info(f"NUM_ROUNDS: {NUM_ROUNDS}")

    history = fl.server.start_server(
        server_address="0.0.0.0:18080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=get_strategy(),
    )

    save_history(history)
