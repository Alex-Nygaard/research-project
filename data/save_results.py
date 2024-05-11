import json
import os
from flwr.server.app import History
from logger.logger import LOG_DIR


def save_history(history: History, filename: str = "history-data.json"):
    full_path = os.path.join(LOG_DIR, filename)

    with open(full_path, "w") as json_file:
        json.dump(history, json_file, default=lambda o: o.__dict__)
