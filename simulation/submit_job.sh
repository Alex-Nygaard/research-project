#!/bin/bash

#SBATCH --job-name="sim"
#SBATCH --output=out.sim.%j.out
#SBATCH --error=err.sim.%j.err
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=education-eemcs-courses-cse3000

module load 2023r1
module load openmpi
module load python
module load py-pip

source ~/venv/bin/activate
pip list >> pip_list.txt
srun python sim.py
echo "Done" >> output.txt
deactivate
