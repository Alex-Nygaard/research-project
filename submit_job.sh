#!/bin/bash

#SBATCH --job-name="sim"
### #SBATCH --output=out.sim.%j.out
### #SBATCH --error=err.sim.%j.err
#SBATCH --partition=compute
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=0
#SBATCH --mem-per-cpu=2G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "logs/run_$SLURM_JOB_ID"

option="simulation"
client_variation="mid"
data_variation="mid"

for arg in "$@"; do
    case $arg in
        --option=*) option="${arg#*=}" ;;
        --client_variation=*) client_variation="${arg#*=}" ;;
        --data_variation=*) data_variation="${arg#*=}" ;;
        *) echo "Unknown parameter passed: $arg"; exit 1 ;;
    esac
done

echo "Running main.py with option=$option, client_variation=$client_variation, data_variation=$data_variation"

conda activate new_env
srun python main.py --option="$option" --client_variation="$client_variation" --data_variation="$data_variation" > "logs/run_$SLURM_JOB_ID/output.log" 2>&1  || echo "Exit with error code $? (suppressed)"
conda deactivate
