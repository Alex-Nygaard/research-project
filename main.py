import os
import asyncio
import shutil
from time import sleep
import flwr as fl

from client.client import FlowerClient
from config.run_config import RunConfig
from config.constants import LOG_DIR, RUN_ID, DATASET
from config.structure import create_output_structure
from data.noniid_dataset_preparation import _download_data
from utils.args import get_base_parser
from utils.run_counter import increment_run_counter
from logger.logger import get_logger

create_output_structure()

logger = get_logger("main")
fl.common.logger.configure(
    identifier="FlowerLog", filename=os.path.join(LOG_DIR, "flwr.log")
)


async def main(config: RunConfig):
    """
    Main function to handle simulation or deployment based on the provided configuration.

    Parameters:
    config (RunConfig): Configuration object containing parameters for running either
                        simulation or deployment.

    This function handles two main operations:
    1. Simulation: Generates client configurations and starts the simulation.
    2. Deployment: Generates or uses provided client configurations, downloads necessary data,
                   and starts server and client tasks for deployment.

    After completing the main tasks, the function writes the configuration to a JSON file
    and increments the run counter.
    """
    if config.option == "simulation":
        logger.info("Starting simulation...")
        logger.info("Generating client config...")
        client_config_file = FlowerClient.generate_simulation_clients(
            num_clients=config.num_clients,
            output_path=os.path.join(LOG_DIR, "clients.json"),
            trace_file=config.trace_file,
            batch_size=config.batch_size,
            local_epochs=config.local_epochs,
            data_volume=config.data_volume,
            data_labels=config.data_labels,
        )
        logger.info(f"Client config generated, saved to {client_config_file}.")
        config.client_config_file = client_config_file
        simulation_task = run_simulation(config)
        logger.info("Simulation task started.")
        await asyncio.gather(simulation_task)
    elif config.option == "deployment":
        logger.info("Starting deployment...")

        if config.trace_file == "":
            logger.info("Generating client config...")
            client_config_file = FlowerClient.generate_deployment_clients(
                config.num_clients, output_path=os.path.join(LOG_DIR, "clients.json")
            )
            logger.info(f"Client config generated, saved to {client_config_file}.")
            config.client_config_file = client_config_file
        else:
            logger.info("Tracefile given, setting client config to it.")
            shutil.copy(config.trace_file, os.path.join(LOG_DIR, "clients.json"))
            config.client_config_file = config.trace_file
            logger.info(f"Downloading dataset '{DATASET}'...")
            _download_data(DATASET, download=True)

        tasks = []
        if config.server_address == "":
            # If server address is not given, start a local server
            config.server_address = "127.0.0.1:5040"
            server_task = asyncio.create_task(run_server(config))
            tasks.append(server_task)
            logger.info("Server task started. Waiting 10 seconds to start clients...")
            await asyncio.sleep(10)

        for i in range(config.num_clients):
            client_task = asyncio.create_task(run_client(i, config))
            tasks.append(client_task)
            logger.info("Client %s task started.", i)
            await asyncio.sleep(0.5)
        logger.info("All client tasks started.")
        await asyncio.gather(*tasks)

    logger.info("Main finished. Writing to disk...")
    config.write_to_json(LOG_DIR, "run_config.json")
    increment_run_counter()


async def run_client(cid: int, config: RunConfig):
    """
    Run a client process with the specified configuration in a subprocess.

    Parameters:
    cid (int): Client ID to uniquely identify the client instance.
    config (RunConfig): Configuration object containing parameters for running the client.
    """
    cmd = [
        "python",
        "deploy_client.py",
        f"--cid={cid}",
        f"--num_clients={config.num_clients}",
        f"--trace_file={config.trace_file}",
        f"--client_config_file={config.client_config_file}",
        f"--server_address={config.server_address}",
        f"--batch_size={config.batch_size}",
        f"--local_epochs={config.local_epochs}",
        f"--data_volume={config.data_volume}",
        f"--data_labels={config.data_labels}",
    ]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


async def run_server(config: RunConfig):
    """
    Run a server process with the specified configuration in a subprocess.

    Parameters:
    config (RunConfig): Configuration object containing parameters for running the server.
    """
    cmd = [
        "python",
        "deploy_server.py",
        f"--num_clients={config.num_clients}",
        f"--server_address={config.server_address}",
        f"--batch_size={config.batch_size}",
        f"--local_epochs={config.local_epochs}",
        f"--data_volume={config.data_volume}",
        f"--data_labels={config.data_labels}",
    ]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


async def run_simulation(config: RunConfig):
    """
    Run a simulation process with the specified configuration in a subprocess.

    Parameters:
    config (RunConfig): Configuration object containing parameters for running the simulation.
    """
    cmd = [
        "python",
        "run_simulation.py",
        f"--num_clients={config.num_clients}",
        f"--trace_file={config.trace_file}",
        f"--client_config_file={config.client_config_file}",
        f"--batch_size={config.batch_size}",
        f"--local_epochs={config.local_epochs}",
        f"--data_volume={config.data_volume}",
        f"--data_labels={config.data_labels}",
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
    num_clients = args.num_clients
    option = args.option
    trace_file = args.trace_file
    # client_config_file = args.client_config_file
    server_address = args.server_address
    batch_size = args.batch_size
    local_epochs = args.local_epochs
    data_volume = args.data_volume
    data_labels = args.data_labels

    run_config = RunConfig(
        run_id=RUN_ID,
        num_clients=num_clients,
        option=option,
        trace_file=trace_file,
        # client_config_file=client_config_file,
        server_address=server_address,
        batch_size=batch_size,
        local_epochs=local_epochs,
        data_volume=data_volume,
        data_labels=data_labels,
    )
    logger.info("Run config: %s", run_config)

    asyncio.run(main(run_config))
