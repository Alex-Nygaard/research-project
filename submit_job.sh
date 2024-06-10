#!/bin/bash

#SBATCH --job-name="sim"
#SBATCH --partition=gpu-v100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "logs/run_$SLURM_JOB_ID"

option="simulation"
batch_size="iid"
local_epochs="iid"
data_volume="iid"
data_labels="iid"
num_clients=100
trace_file="clients/testing_clients.json"

for arg in "$@"; do
    case $arg in
        --option=*) option="${arg#*=}" ;;
        --batch_size=*) batch_size="${arg#*=}" ;;
        --local_epochs=*) local_epochs="${arg#*=}" ;;
        --data_volume=*) data_volume="${arg#*=}" ;;
        --data_labels=*) data_labels="${arg#*=}" ;;
        --num_clients=*) num_clients="${arg#*=}" ;;
        --trace_file=*) trace_file="${arg#*=}" ;;
        *) echo "Unknown parameter passed: $arg"; exit 1 ;;
    esac
done

echo "Running main.py with option=$option, batch_size=$batch_size, local_epochs=$local_epochs, data_volume=$data_volume, data_labels=$data_labels, num_clients=$num_clients, trace_file=$trace_file"

conda activate new_env
srun python main.py --option="$option" --batch_size="$batch_size" --local_epochs="$local_epochs" --data_volume="$data_volume" --data_labels="$data_labels" --num_clients="$num_clients" --trace_file="$trace_file" > "logs/run_$SLURM_JOB_ID/output.log" 2>&1 || echo "Exit with error code $? (suppressed)"
conda deactivate

rm -rf "/tmp/$SLURM_JOB_ID"