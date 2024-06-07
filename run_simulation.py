import os
import flwr as fl

from strategy.strategy import get_strategy
from client.client import get_client_fn
from config.constants import NUM_ROUNDS, LOG_DIR
from logger.logger import get_logger
from data.save_results import save_history
from utils.args import get_base_parser

if __name__ == "__main__":
    parser = get_base_parser("Running a simulation.")

    args = parser.parse_args()
    num_clients = args.num_clients
    batch_size = args.batch_size
    local_epochs = args.local_epochs
    data_volume = args.data_volume
    data_labels = args.data_labels
    trace_file = args.trace_file

    logger = get_logger("simulation.run")
    fl.common.logger.configure(
        identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
    )

    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    num_gpus = int(os.environ.get("SLURM_GPUS_PER_TASK", 0))

    logger.info(
        "RUNNING SIMULATION with batch_size=%s,local_epochs=%s,data_volume=%s,data_labels=%s,num_clients=%s,trace_file=%s,num_cpus=%s,num_gpus=%s",
        batch_size,
        local_epochs,
        data_volume,
        data_labels,
        num_clients,
        trace_file,
        num_cpus,
        num_gpus,
    )

    history = fl.simulation.start_simulation(
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        client_fn=get_client_fn(
            num_clients=num_clients,
            option="simulation",
            trace_file=trace_file,
            batch_sizes=batch_size,
            local_epochs=local_epochs,
            data_volume=data_volume,
            data_labels=data_labels,
        ),
        strategy=get_strategy(),
        ray_init_args={
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
        },
    )

    save_history(history)
