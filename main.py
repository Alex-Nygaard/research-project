import os
import asyncio
import argparse
from time import sleep
import flwr as fl

from client.attribute import Attribute
from config.run_config import RunConfig
from config.constants import LOG_DIR, RUN_ID
from config.structure import create_output_structure
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
        server_task = run_server()
        logger.info("Server task started. Waiting 10 seconds to start clients...")
        sleep(10)
        client_tasks = []
        for i in range(Attribute("num_clients", config.client_variation).generate()):
            client_tasks.append(run_client(i, config))
            sleep(0.5)
        logger.info("Client tasks (%s) started.", len(client_tasks))
        await asyncio.gather(server_task, *client_tasks)

    logger.info("Main finished. Writing to disk...")
    config.write_to_json(LOG_DIR, "run_config.json")
    increment_run_counter()


async def run_client(partition_id: int, config: RunConfig):
    logger.info("Starting client %s.", partition_id)
    cmd = [
        "python",
        "deploy_client.py",
        f"--partition_id={partition_id}",
        f"--client_variation={config.client_variation}",
        f"--data_variation={config.data_variation}",
    ]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


async def run_server():
    # Prepare the command
    cmd = ["python", "deploy_server.py"]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


async def run_simulation(config: RunConfig):
    cmd = [
        "python",
        "run_simulation.py",
        f"--client_variation={config.client_variation}",
        f"--data_variation={config.data_variation}",
    ]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


if __name__ == "__main__":
    logger.info("Starting main.")
    sleep(0.5)
    parser = argparse.ArgumentParser(description="FL simulation and deployment runner.")
    parser.add_argument(
        "--option",
        type=str,
        choices=["simulation", "deployment"],
        default="deployment",
        nargs="?",
        help="Either 'simulation' or 'deployment'",
    )
    parser.add_argument(
        "--client_variation",
        type=str,
        choices=["low", "mid", "high"],
        default="mid",
        nargs="?",
        help="Client attribute variation.",
    )
    parser.add_argument(
        "--data_variation",
        type=str,
        choices=["low", "mid", "high"],
        default="mid",
        nargs="?",
        help="Data attribute variation.",
    )

    args = parser.parse_args()
    option = args.option
    client_variation = args.client_variation
    data_variation = args.data_variation

    run_config = RunConfig(RUN_ID, option, client_variation, data_variation)
    logger.info("Run config: %s", run_config)

    asyncio.run(main(run_config))
