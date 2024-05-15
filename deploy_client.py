import os
import flwr as fl
import argparse

from client.client import get_client_fn
from logger.logger import get_logger
from config.constants import LOG_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running a client.")
    parser.add_argument("--partition_id", type=int, help="Partition ID", required=True)
    parser.add_argument("--client_variation", type=str, help="Client variation")
    parser.add_argument("--data_variation", type=str, help="Client variation")

    args = parser.parse_args()
    partition_id = args.partition_id
    client_variation = args.client_variation
    data_variation = args.data_variation

    logger = get_logger(f"client-{partition_id}.deploy")
    fl.common.logger.configure(
        identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
    )

    logger.info(f"Starting client with partition_id={partition_id}")

    fl.client.start_client(
        server_address="127.0.0.1:5040",
        client_fn=get_client_fn(
            client_variation, data_variation, deployment_id=partition_id
        ),
    )
