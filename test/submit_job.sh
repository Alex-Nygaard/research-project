#!/bin/bash

#SBATCH --job-name="testname1"
#SBATCH --output=testname1.%j.out
#SBATCH --error=testname1.%j.err
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load python
module load py-pip

source ~/venv/bin/activate

srun python test_python.py

deactivate