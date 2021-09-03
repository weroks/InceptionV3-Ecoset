#!/bin/bash -l
#SBATCH -D ./

#SBATCH -o out/fe_RMSprop.%j
#SBATCH -e out/fe_RMSprop.%j
#SBATCH -J fe_RMSprop

#SBATCH -t 2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=0
#SBATCH --cpus-per-task=18

# --- start from a clean state and load necessary environment modules --- #
module purge
module load anaconda
module load cuda/11.2
module load pytorch/gpu-cuda-11.2/1.8.1
module load gcc/10
module load openmpi/4
module load horovod-pytorch-1.8.1/gpu-cuda-11.2/0.21.0
srun python ./inception.py --batch=256 --lr=0.01 --fixed_feat --optim RMSprop