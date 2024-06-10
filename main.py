import os
import asyncio
from time import sleep
import flwr as fl

from client.client import FlowerClient
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
        logger.info("Generating client config...")
        trace_file = FlowerClient.generate_simulation_clients(
            num_clients=config.num_clients,
            output_path=os.path.join(LOG_DIR, "clients.json"),
            trace_file=config.trace_file,
            batch_size=config.batch_size,
            local_epochs=config.local_epochs,
            data_volume=config.data_volume,
            data_labels=config.data_labels,
        )
        config.trace_file = trace_file
        simulation_task = run_simulation(config)
        logger.info("Simulation task started.")
        await asyncio.gather(simulation_task)
    elif config.option == "deployment":
        logger.info("Starting deployment...")

        logger.info("Generating client config...")
        output_dirs = [
            os.path.join(LOG_DIR, "clients.json"),
            os.path.join("client", "testing_clients.json"),
        ]
        FlowerClient.generate_deployment_clients(config.num_clients, output_dirs)
        config.trace_file = output_dirs[0]
        logger.info(f"RunConfig trace file set to {config.trace_file}.")

        server_task = asyncio.create_task(run_server(config))
        logger.info("Server task started. Waiting 10 seconds to start clients...")
        await asyncio.sleep(10)
        client_tasks = []
        for i in range(config.num_clients):
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
        f"--num_clients={config.num_clients}",
        f"--trace_file={config.trace_file}",
        f"--batch_size={config.batch_size}",
        f"--local_epochs={config.local_epochs}",
        f"--data_volume={config.data_volume}",
        f"--data_labels={config.data_labels}",
    ]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


async def run_server(config: RunConfig):
    # Prepare the command
    cmd = [
        "python",
        "deploy_server.py",
        f"--num_clients={config.num_clients}",
        f"--batch_size={config.batch_size}",
        f"--local_epochs={config.local_epochs}",
        f"--data_volume={config.data_volume}",
        f"--data_labels={config.data_labels}",
    ]
    process = await asyncio.create_subprocess_exec(*cmd, cwd=os.getcwd())
    await process.wait()


async def run_simulation(config: RunConfig):
    cmd = [
        "python",
        "run_simulation.py",
        f"--num_clients={config.num_clients}",
        f"--trace_file={config.trace_file}",
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
    batch_size = args.batch_size
    local_epochs = args.local_epochs
    data_volume = args.data_volume
    data_labels = args.data_labels

    run_config = RunConfig(
        run_id=RUN_ID,
        num_clients=num_clients,
        option=option,
        trace_file=trace_file,
        batch_size=batch_size,
        local_epochs=local_epochs,
        data_volume=data_volume,
        data_labels=data_labels,
    )
    logger.info("Run config: %s", run_config)

    asyncio.run(main(run_config))
