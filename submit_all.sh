#!/bin/bash


# Base deployment
sbatch submit_job.sh --option="deployment"

# Base simulation
sbatch submit_job.sh --option="simulation"

# Varied simulations
sbatch submit_job.sh --option="simulation" --resources="low"
sbatch submit_job.sh --option="simulation" --resources="high"

sbatch submit_job.sh --option="simulation" --variability="low"
sbatch submit_job.sh --option="simulation" --variability="high"

sbatch submit_job.sh --option="simulation" --concentration="low"
sbatch submit_job.sh --option="simulation" --concentration="high"

#sbatch submit_job.sh --option="simulation" --distribution="low"
#sbatch submit_job.sh --option="simulation" --distribution="high"
