#!/bin/bash -l
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # If using both GPUs of a node
#SBATCH --cpus-per-task=16
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=100000

# --- start from a clean state and load necessary environment modules --- #
module purge
module load anaconda
module load cuda/11.2
module load pytorch/gpu-cuda-11.2/1.8.1
module load gcc/10
module load openmpi/4
module load horovod-pytorch-1.8.1/gpu-cuda-11.2/0.21.0
srun python ./vgg_nohvd.py --batch=44
