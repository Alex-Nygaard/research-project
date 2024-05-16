#!/bin/bash


# Submit deployment job
sbatch submit_job.sh --option="deployment"

# Submit simulation jobs
sbatch submit_job.sh --option="simulation" --client_variation="low" --data_variation="mid"
sbatch submit_job.sh --option="simulation" --client_variation="high" --data_variation="mid"

sbatch submit_job.sh --option="simulation" --client_variation="mid" --data_variation="low"
sbatch submit_job.sh --option="simulation" --client_variation="mid" --data_variation="high"

sbatch submit_job.sh --option="simulation" --client_variation="mid" --data_variation="mid"
