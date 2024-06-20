import os
import argparse

from client.client import FlowerClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    num_clients = args.num_clients
    seed = args.seed

    output_path = os.path.join("logs", "clients", f"clients{num_clients}_{seed}.json")

    print(f"Generating client attributes for {num_clients} clients (seed={seed})...")
    FlowerClient.generate_deployment_clients(num_clients, output_path, seed=seed)
    print(f"Client attributes written to {output_path}")
