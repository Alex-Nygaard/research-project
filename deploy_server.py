import asyncio
import os
import flwr as fl
from strategy.strategy import get_strategy
from config.constants import NUM_ROUNDS
from logger.logger import get_logger

logger = get_logger("server.run")


# async def run_server():
#     # Prepare the command
#     cmd = ["python", "server/run.py"]
#     process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
#     await process.wait()


if __name__ == "__main__":
    logger.info("Starting server.")
    logger.info(f"NUM_ROUNDS: {NUM_ROUNDS}")

    fl.server.start_server(
        server_address="0.0.0.0:18080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=get_strategy(),
    )
