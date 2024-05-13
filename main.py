import os
import asyncio
import argparse
from time import sleep
import flwr as fl

from config.run_config import RunConfig
from config.constants import NUM_CLIENTS, LOG_DIR, RUN_ID
from config.structure import create_output_structure
from client.client import FlowerClient
from utils.run_counter import increment_run_counter, read_run_counter
from logger.logger import get_logger

create_output_structure()

logger = get_logger("main")
fl.common.logger.configure(
    identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
)


async def main(config: RunConfig):

    # clients = FlowerClient.generate_clients(
    #     NUM_CLIENTS, config.client_variation, config.data_variation
    # )
    # FlowerClient.write_many(clients, LOG_DIR, "clients.csv")

    if config.option == "simulation":
        logger.info("Starting simulation...")
        simulation_task = run_simulation(config)
        logger.info("Simulation task started.")
        await asyncio.gather(simulation_task)
    elif config.option == "deployment":
        logger.info("Starting deployment...")
        server_task = run_server()
        logger.info("Server task started. Waiting 5 seconds to start clients...")
        sleep(5)
        client_tasks = [run_client(i, config) for i in range(NUM_CLIENTS)]
        logger.info("Client tasks (%s) started.", NUM_CLIENTS)
        await asyncio.gather(server_task, *client_tasks)

    config.write_to_json(LOG_DIR, "run_config.json")
    increment_run_counter()


async def run_client(partition_id: int, config: RunConfig):
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
        "option",
        type=str,
        choices=["simulation", "deployment"],
        default="simulation",
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
