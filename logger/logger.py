import logging
import os
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists("logs"):
    os.makedirs("logs")
os.makedirs(f"logs/{current_time}")

logging.basicConfig(
    filename=f"logs/{current_time}/output.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
