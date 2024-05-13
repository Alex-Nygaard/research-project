import os

from config.constants import DATA_SAVE_PATH, LOG_DIR
from client.client import FlowerClient


def create_output_structure():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)
    run_counter_path = os.path.join("logs", "run_counter.txt")
    if not os.path.exists(run_counter_path):
        # Open the file in write mode which will create the file if it doesn't exist
        with open(run_counter_path, "w") as file:
            file.write("0")
            file.flush()

    clients_csv_path = os.path.join(LOG_DIR, "clients.csv")
    with open(clients_csv_path, "w") as file:
        file.write(",".join(FlowerClient.columns) + "\n")
        file.flush()
