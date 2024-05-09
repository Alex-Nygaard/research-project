import os
from deployment.run import start_server
from logger.logger import get_logger

logger = get_logger("main")


if __name__ == "__main__":
    logger.info("Running main.")
    start_server()
