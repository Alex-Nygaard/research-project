import flwr as fl
import argparse

from client.client import FlowerClient
from logger.logger import get_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running a client.")
    parser.add_argument("--partition-id", type=int, help="Partition ID", required=True)
    args = parser.parse_args()
    cid = args.partition_id

    logger = get_logger(f"client-{cid}.deploy")

    logger.info(f"Starting client with partition_id={cid}")

    fl.client.start_client(
        server_address="127.0.0.1:18080",
        client=FlowerClient(cid).to_client(),
    )
