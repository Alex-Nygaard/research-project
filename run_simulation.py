import os
import flwr as fl

from client.attribute import Attribute
from strategy.strategy import get_strategy
from client.client import get_client_fn
from config.constants import NUM_ROUNDS, LOG_DIR
from logger.logger import get_logger
from data.save_results import save_history
from utils.args import get_base_parser

if __name__ == "__main__":
    parser = get_base_parser("Running a simulation.")

    args = parser.parse_args()
    resources = args.resources
    concentration = args.concentration
    variability = args.variability
    quality = args.quality

    logger = get_logger("simulation.run")
    fl.common.logger.configure(
        identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
    )

    num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    num_gpus = int(os.environ.get("SLURM_GPUS_PER_TASK", 0))

    logger.info(
        "RUNNING SIMULATION with resources=%s,concentration=%s,variability=%s,quality=%s,num_cpus=%s,num_gpus=%s",
        resources,
        concentration,
        variability,
        quality,
        num_cpus,
        num_gpus,
    )

    history = fl.simulation.start_simulation(
        num_clients=Attribute("num_clients", concentration).generate(),
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        client_fn=get_client_fn(resources, concentration, variability, quality),
        strategy=get_strategy(),
        ray_init_args={
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
        },
    )

    save_history(history)
