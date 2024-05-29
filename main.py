import os
import asyncio
import argparse
from time import sleep
import flwr as fl

from client.attribute import Attribute
from config.run_config import RunConfig
from config.constants import LOG_DIR, RUN_ID
from config.structure import create_output_structure
from utils.args import get_base_parser
from utils.run_counter import increment_run_counter
from logger.logger import get_logger

create_output_structure()

logger = get_logger("main")
fl.common.logger.configure(
    identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
)


async def main(config: RunConfig):

    if config.option == "simulation":
        logger.info("Starting simulation...")
        simulation_task = run_simulation(config)
        logger.info("Simulation task started.")
        await asyncio.gather(simulation_task)
    elif config.option == "deployment":
        logger.info("Starting deployment...")
        server_task = asyncio.create_task(run_server(config))
        logger.info("Server task started. Waiting 10 seconds to start clients...")
        await asyncio.sleep(10)
        client_tasks = []
        for i in range(Attribute("num_clients", config.concentration).generate()):
            client_task = asyncio.create_task(run_client(i, config))
            client_tasks.append(client_task)
            logger.info("Client %s task started.", i)
            await asyncio.sleep(0.5)
        logger.info("All client tasks (%s) started.", len(client_tasks))
        await asyncio.gather(server_task, *client_tasks)

    logger.info("Main finished. Writing to disk...")
    config.write_to_json(LOG_DIR, "run_config.json")
    increment_run_counter()


async def run_client(cid: int, config: RunConfig):
    # logger.info("Starting client %s.", cid)
    cmd = [
        "python",
        "deploy_client.py",
        f"--cid={cid}",
        f"--resources={config.resources}",
        f"--concentration={config.concentration}",
        f"--variability={config.variability}",
        f"--distribution={config.distribution}",
    ]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


async def run_server(config: RunConfig):
    # Prepare the command
    cmd = [
        "python",
        "deploy_server.py",
        f"--resources={config.resources}",
        f"--concentration={config.concentration}",
        f"--variability={config.variability}",
        f"--distribution={config.distribution}",
    ]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


async def run_simulation(config: RunConfig):
    cmd = [
        "python",
        "run_simulation.py",
        f"--resources={config.resources}",
        f"--concentration={config.concentration}",
        f"--variability={config.variability}",
        f"--distribution={config.distribution}",
    ]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


if __name__ == "__main__":
    logger.info("Starting main.")
    sleep(0.5)
    parser = get_base_parser("FL simulation and deployment runner.")
    parser.add_argument(
        "--option",
        type=str,
        choices=["simulation", "deployment"],
        default="deployment",
        nargs="?",
        help="Either 'simulation' or 'deployment'",
    )

    args = parser.parse_args()
    option = args.option
    resources = args.resources
    concentration = args.concentration
    variability = args.variability
    distribution = args.distribution

    run_config = RunConfig(
        RUN_ID, option, resources, concentration, variability, distribution
    )
    logger.info("Run config: %s", run_config)

    asyncio.run(main(run_config))
