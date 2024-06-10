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
    num_clients = args.num_clients
    trace_file = args.trace_file
    batch_size = args.batch_size
    local_epochs = args.local_epochs
    data_volume = args.data_volume
    data_labels = args.data_labels

    logger = get_logger(f"client-{cid}.deploy")
    fl.common.logger.configure(
        identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
    )

    logger.info(f"Starting client with partition_id={cid}")

    fl.client.start_client(
        server_address="127.0.0.1:5040",
        client_fn=get_client_fn(
            num_clients=num_clients,
            option="deployment",
            trace_file=trace_file,
            batch_sizes=batch_size,
            local_epochs=local_epochs,
            data_volume=data_volume,
            data_labels=data_labels,
            deployment_id=cid,
        ),
    )
