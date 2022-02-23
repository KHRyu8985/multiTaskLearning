
# We assume running this from the script directory
job_directory=$PWD/.job
out_directory=$PWD/.out
mkdir -p ${job_directory}
mkdir -p ${out_directory}

exp_names=(
    'eval_TRUESHARE_SPLIT'
    'eval_MHUSHARE_SPLIT'
    'eval_MHUSHARE_ALL'
    'eval_TRUESHARE_ALL'
    'eval_STL_shoulder'
    'eval_TFLEARN_wrist_to_shoulder'
    'eval_TFLEARN_ankle_elbow_to_shoulder'
    )

cmds=(
    'python evaluate_v2.py --datasets ankle_elbow shoulder wrist --datasplit Val --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/TRUESHARE_SPLIT_modlIIIIIIVV_ankle_elbow_shoulder_wrist --resultdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/results --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare split split --device cuda:0 --numworkers 1'
    'python evaluate_v2.py --datasets ankle_elbow shoulder wrist --datasplit Val --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/MHUSHARE_SPLIT_modlYYYYYYVV_ankle_elbow_shoulder_wrist --resultdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/results --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare split split --device cuda:0 --numworkers 1'
    'python evaluate_v2.py --datasets ankle_elbow shoulder wrist --datasplit Val --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/MHUSHARE_ALL_modlYYYYYYYY_ankle_elbow_shoulder_wrist --resultdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/results --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare --device cuda:0 --numworkers 1'
    'python evaluate_v2.py --datasets ankle_elbow shoulder wrist --datasplit Val --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/TRUESHARE_ALL_modlIIIIIIII_ankle_elbow_shoulder_wrist --resultdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/results --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare trueshare trueshare --device cuda:0 --numworkers 1'
    'python evaluate_v2.py --datasets shoulder --datasplit Val --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/STL_SPLIT_modlVVVVVVVV_shoulder --resultdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/results --blockstructures split split split split split split split split --device cuda:0 --numworkers 1'
    'python evaluate_v2.py --datasets shoulder --datasplit Val --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/TFLEARN_STL_wrist_lr1e-5_modlVVVVVVVV_shoulder --resultdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/results --blockstructures split split split split split split split split --device cuda:0 --numworkers 1'
    'python evaluate_v2.py --datasets shoulder --datasplit Val --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/TFLEARN_STL_ankle_elbow_lr1e-5_modlVVVVVVVV_shoulder --resultdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/results --blockstructures split split split split split split split split --device cuda:0 --numworkers 1'
    )


for i in ${!exp_names[@]}; do
    
    exp_name="${exp_names[$i]}"

    job_file="${job_directory}/${exp_name}.job"
    curdate=$(date +'%Y_%m_%d__%H_%M_%N')

    echo "#!/bin/bash
#SBATCH --job-name=${exp_name}.job
#SBATCH --output=${out_directory}/${exp_name}.out
#SBATCH --error=${out_directory}/${exp_name}.err
#SBATCH --time=04:00
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:12GB|GPU_MEM:32GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=calkan@stanford.edu
source /home/users/${USER}/.bashrc
conda activate multitaskrecon
module load cudnn/7.6.5
cd /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning
${cmds[$i]}" > $job_file
    sbatch $job_file

done
