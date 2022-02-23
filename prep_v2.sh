#!/bin/bash
#SBATCH --job-name=prep_%j
#SBATCH --output=prep_%j.out
#SBATCH --error=prep_%j.err
#SBATCH --time=1-00:00
#SBATCH -c 5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kanghyun@stanford.edu

source /home/users/${USER}/.bashrc

module load cuda/11.2
module load cudnn/8.1.1.33

conda activate mtl
cd /home/groups/shreyasv/multiTaskLearning
python prep_v2.py