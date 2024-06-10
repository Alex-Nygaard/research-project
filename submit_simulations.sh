#!/bin/bash

# Blind simulation
sbatch submit_job.sh --option="simulation" --num_clients=50 --trace_file="client/testing_clients.json"

# Batch size simulations
sbatch submit_job.sh --option="simulation" --num_clients=50 --trace_file="client/testing_clients.json" --batch_size="noniid"
sbatch submit_job.sh --option="simulation" --num_clients=50 --trace_file="client/testing_clients.json" --local_epochs="noniid"
sbatch submit_job.sh --option="simulation" --num_clients=50 --trace_file="client/testing_clients.json" --data_volume="noniid"
sbatch submit_job.sh --option="simulation" --num_clients=50 --trace_file="client/testing_clients.json" --data_labels="noniid"

# Realistic simulations
sbatch submit_job.sh --option="simulation" --num_clients=50 --trace_file="client/testing_clients.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
