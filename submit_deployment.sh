#!/bin/bash


# Base deployment
sbatch submit_job.sh --option="deployment" --num_clients=100 --batch_size="noniid" --local_epochs="noniid" --data_volume="noniid" --data_labels="noniid"
