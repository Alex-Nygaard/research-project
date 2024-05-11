#!/bin/bash

#SBATCH --job-name="simulation"
#SBATCH --output=out.sim.%j.out
#SBATCH --error=err.sim.%j.err
#SBATCH --partition=gpu-v100
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load cuda/11.6
module load python
module load py-pip

source ~/venv/bin/activate
srun python main.py simulation
deactivate
