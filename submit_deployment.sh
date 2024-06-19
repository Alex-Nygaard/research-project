#!/bin/bash

# Base deployment
sbatch submit_job.sh --option="deployment" --num_clients=40 --trace_file="logs/clients/clients40_1001.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
sbatch submit_job.sh --option="deployment" --num_clients=40 --trace_file="logs/clients/clients40_1002.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
sbatch submit_job.sh --option="deployment" --num_clients=40 --trace_file="logs/clients/clients40_1003.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
sbatch submit_job.sh --option="deployment" --num_clients=40 --trace_file="logs/clients/clients40_1004.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
sbatch submit_job.sh --option="deployment" --num_clients=40 --trace_file="logs/clients/clients40_1005.json" --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"

