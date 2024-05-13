#!/bin/bash

#SBATCH --job-name="sim"
### #SBATCH --output=out.sim.%j.out
### #SBATCH --error=err.sim.%j.err
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
### #SBATCH --gpus-per-task=0
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -f logs/run_counter.txt ]; then
    echo "0" > logs/run_counter.txt
fi

RUN_NUMBER=$(cat logs/run_counter.txt)
mkdir -p "logs/run_${RUN_NUMBER}"

conda activate new_env
srun python main.py simulation > "logs/run_${RUN_NUMBER}/output.log" 2>&1
conda deactivate
