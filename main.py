import os
import asyncio
import argparse
from time import sleep

from config.constants import NUM_CLIENTS
from logger.logger import get_logger

logger = get_logger("main")


async def main(option: str):
    logger.info("Running main with option=%s", option)
    if option == "simulation":
        pass
    elif option == "deployment":
        server_task = run_server()
        logger.info("Server task started. Waiting 5 seconds to start clients...")
        sleep(10)
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
