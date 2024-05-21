#!/bin/bash


# Submit deployment job
sbatch submit_job.sh --option="deployment"

# Submit simulation jobs
#sbatch submit_job.sh --option="simulation" --concentration="low"
#sbatch submit_job.sh --option="simulation" --concentration="mid"
#sbatch submit_job.sh --option="simulation" --concentration="high"
