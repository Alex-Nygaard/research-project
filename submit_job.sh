#!/bin/bash

#SBATCH --job-name="sim"
### #SBATCH --output=out.sim.%j.out
### #SBATCH --error=err.sim.%j.err
#SBATCH --partition=gpu-a100
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "logs/run_$SLURM_JOB_ID"

option="simulation"
resources="mid"
concentration="mid"
variability="mid"
quality="mid"

for arg in "$@"; do
    case $arg in
        --option=*) option="${arg#*=}" ;;
        --resources=*) resources="${arg#*=}" ;;
        --concentration=*) concentration="${arg#*=}" ;;
        --variability=*) variability="${arg#*=}" ;;
        --quality=*) quality="${arg#*=}" ;;
        *) echo "Unknown parameter passed: $arg"; exit 1 ;;
    esac
done

echo "Running main.py with option=$option, resources=$resources, concentration=$concentration, variability=$variability, quality=$quality"

conda activate new_env
srun python main.py --option="$option" --resources="$resources" --concentration="$concentration" --variability="$variability" --quality="$quality" > "logs/run_$SLURM_JOB_ID/output.log" 2>&1  || echo "Exit with error code $? (suppressed)"
conda deactivate
