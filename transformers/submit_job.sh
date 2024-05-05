#!/bin/bash

#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --job-name="sim"
#SBATCH --output=out.sim.%j.out
#SBATCH --error=err.sim.%j.err
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --mem-per-cpu=3G

module load 2023r1
module load openmpi
module load python
module load py-pip
module load cuda/11.6

source ~/venv/bin/activate

srun run.sh

deactivate
