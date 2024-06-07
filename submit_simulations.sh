#!/bin/bash

# Blind simulation
sbatch submit_job.sh --option="simulation" --num_clients=100

# Batch size simulations
sbatch submit_job.sh --option="simulation" --num_clients=100 --batch_size="noniid"
sbatch submit_job.sh --option="simulation" --num_clients=100 --local_epochs="noniid"
sbatch submit_job.sh --option="simulation" --num_clients=100 --data_volume="noniid"
sbatch submit_job.sh --option="simulation" --num_clients=100 --data_labels="noniid"

# Realistic simulations
sbatch submit_job.sh --option="simulation" --num_clients=100 --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
