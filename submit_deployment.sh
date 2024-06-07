#!/bin/bash


# Base deployment
sbatch submit_job.sh --option="deployment"

# Blind simulation
sbatch submit_job.sh --option="simulation"

# Batch size simulations
sbatch submit_job.sh --option="simulation" --batch_size="non-iid"
sbatch submit_job.sh --option="simulation" --local_epochs="non-iid"
sbatch submit_job.sh --option="simulation" --data_volume="non-iid"
sbatch submit_job.sh --option="simulation" --data_labels="non-iid"

# Realistic simulations
sbatch submit_job.sh --option="simulation" --batch_size="non-iid" --local_epochs="non-iid" --data_volume="non-iid" --data_labels="non-iid"
