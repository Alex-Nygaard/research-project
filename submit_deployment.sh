#!/bin/bash

# Base deployment
sbatch submit_job.sh --option="deployment" --num_clients=50 --trace_file="logs/clients/clients1001.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
sbatch submit_job.sh --option="deployment" --num_clients=50 --trace_file="logs/clients/clients1002.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
sbatch submit_job.sh --option="deployment" --num_clients=50 --trace_file="logs/clients/clients1003.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
sbatch submit_job.sh --option="deployment" --num_clients=50 --trace_file="logs/clients/clients1004.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
sbatch submit_job.sh --option="deployment" --num_clients=50 --trace_file="logs/clients/clients1005.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"


#sbatch submit_job.sh --option="deployment" --num_clients=20 --trace_file="logs/clients/clients20_01.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
