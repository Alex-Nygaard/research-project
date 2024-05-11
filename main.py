import os
import asyncio
import argparse
from time import sleep
import flwr as fl

from utils.run_counter import increment_run_counter

run_count = increment_run_counter()

from logger.logger import get_logger
from config.constants import NUM_CLIENTS, LOG_DIR


logger = get_logger("main")
fl.common.logger.configure(
    identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
)


async def main(option: str):
    logger.info("Running main with option=%s", option)
    if option == "simulation":
        simulation_task = run_simulation()
        logger.info("Simulation task started.")
        await asyncio.gather(simulation_task)
    elif option == "deployment":
        server_task = run_server()
        logger.info("Server task started. Waiting 5 seconds to start clients...")
        sleep(5)
        client_tasks = [run_client(i) for i in range(NUM_CLIENTS)]
        logger.info("Client tasks (%s) started.", NUM_CLIENTS)
        await asyncio.gather(server_task, *client_tasks)


async def run_client(partition_id):
    cmd = ["python", "deploy_client.py", f"--partition-id={partition_id}"]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


async def run_server():
    # Prepare the command
    cmd = ["python", "deploy_server.py"]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


async def run_simulation():
    cmd = [
        "python",
        "run_simulation.py",
        "--client_variation=mid",
        "--data_variation=mid",
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
        help="Either 'simulation' or 'deployment'",
    )
    args = parser.parse_args()
    option = args.option

    asyncio.run(main(option))
