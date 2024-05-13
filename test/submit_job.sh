#!/bin/bash

#SBATCH --job-name="testname1"
#SBATCH --output=testname1.%j.out
#SBATCH --error=testname1.%j.err
#SBATCH --partition=compute
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
### #SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
#module load python
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate exp_env
srun python testing.py
conda deactivate