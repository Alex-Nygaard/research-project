import os
import flwr as fl

from client.attribute import Attribute
from strategy.strategy import get_strategy
from config.constants import NUM_ROUNDS, LOG_DIR
from logger.logger import get_logger
from data.save_results import save_history
from utils.args import get_base_parser

if __name__ == "__main__":
    logger = get_logger("server.deploy")
    fl.common.logger.configure(
        identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
    )

    parser = get_base_parser("Deploy an FL server.")
    args = parser.parse_args()
    resources = args.resources
    concentration = args.concentration
    variability = args.variability
    quality = args.quality

    logger.info("Starting server.")
    logger.info(f"NUM_ROUNDS: {NUM_ROUNDS}")

    history = fl.server.start_server(
        server_address="127.0.0.1:5040",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=get_strategy(
            min_available_clients=Attribute("num_clients", concentration).generate()
        ),
    )

    save_history(history)
