import os
import flwr as fl

from client.client import get_client_fn
from logger.logger import get_logger
from config.constants import LOG_DIR
from utils.args import get_base_parser

if __name__ == "__main__":
    parser = get_base_parser("Deploy an FL client.")
    parser.add_argument("--cid", type=int, help="Partition ID", required=True)

    args = parser.parse_args()
    cid = args.cid
    resources = args.resources
    concentration = args.concentration
    variability = args.variability
    distribution = args.distribution

    logger = get_logger(f"client-{cid}.deploy")
    fl.common.logger.configure(
        identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
    )

    logger.info(f"Starting client with partition_id={cid}")

    fl.client.start_client(
        server_address="127.0.0.1:5040",
        client_fn=get_client_fn(
            resources, concentration, variability, distribution, deployment_id=cid
        ),
    )
