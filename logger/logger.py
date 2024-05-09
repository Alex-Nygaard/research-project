import logging
import os
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

LOG_DIR = f"logs/{current_time}"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logging.basicConfig(
    filename=f"{LOG_DIR}/main_output.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
