
# We assume running this from the script directory
job_directory=$PWD/.job
out_directory=$PWD/.out
mkdir -p ${job_directory}
mkdir -p ${out_directory}

exp_names=(
    'TFLEARN_STL_ankle_elbow_s1_lr1e5_shoulder_s1'
    'TFLEARN_STL_ankle_elbow_s1_lr1e5_shoulder_s2'
    'TFLEARN_STL_ankle_elbow_s1_lr1e5_shoulder_s3'
    'TFLEARN_STL_ankle_elbow_s1_lr1e5_shoulder_s4'
    'TFLEARN_STL_ankle_elbow_s1_lr1e5_shoulder_s5'
    'TFLEARN_STL_wrist_s0_lr1e5_shoulder_s1'
    'TFLEARN_STL_wrist_s0_lr1e5_shoulder_s2'
    'TFLEARN_STL_wrist_s0_lr1e5_shoulder_s3'
    'TFLEARN_STL_wrist_s0_lr1e5_shoulder_s4'
    'TFLEARN_STL_wrist_s0_lr1e5_shoulder_s5'
    )

cmds=(
    'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname TFLEARN_STL_ankle_elbow_lr1e-5 --lr 0.00001 --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/STL_SPLIT_modlVVVVVVVV_ankle_elbow/N=578_l1.pt --scarcities 1 --device cuda:0 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname TFLEARN_STL_ankle_elbow_lr1e-5 --lr 0.00001 --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/STL_SPLIT_modlVVVVVVVV_ankle_elbow/N=578_l1.pt --scarcities 2 --device cuda:0 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname TFLEARN_STL_ankle_elbow_lr1e-5 --lr 0.00001 --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/STL_SPLIT_modlVVVVVVVV_ankle_elbow/N=578_l1.pt --scarcities 3 --device cuda:0 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname TFLEARN_STL_ankle_elbow_lr1e-5 --lr 0.00001 --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/STL_SPLIT_modlVVVVVVVV_ankle_elbow/N=578_l1.pt --scarcities 4 --device cuda:0 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname TFLEARN_STL_ankle_elbow_lr1e-5 --lr 0.00001 --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/STL_SPLIT_modlVVVVVVVV_ankle_elbow/N=578_l1.pt --scarcities 5 --device cuda:0 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname TFLEARN_STL_wrist_lr1e-5 --lr 0.00001 --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/STL_SPLIT_modlVVVVVVVV_wrist/N=632_l1.pt --scarcities 1 --device cuda:0 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname TFLEARN_STL_wrist_lr1e-5 --lr 0.00001 --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/STL_SPLIT_modlVVVVVVVV_wrist/N=632_l1.pt --scarcities 2 --device cuda:0 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname TFLEARN_STL_wrist_lr1e-5 --lr 0.00001 --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/STL_SPLIT_modlVVVVVVVV_wrist/N=632_l1.pt --scarcities 3 --device cuda:0 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname TFLEARN_STL_wrist_lr1e-5 --lr 0.00001 --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/STL_SPLIT_modlVVVVVVVV_wrist/N=632_l1.pt --scarcities 4 --device cuda:0 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname TFLEARN_STL_wrist_lr1e-5 --lr 0.00001 --weightsdir /scratch/users/calkan/MTLjoint_tempfiles/multiTaskLearning/models/STL_SPLIT_modlVVVVVVVV_wrist/N=632_l1.pt --scarcities 5 --device cuda:0 --savefreq 1 --numworkers 1'
    )

for i in ${!exp_names[@]}; do
    
    exp_name="${exp_names[$i]}"

    job_file="${job_directory}/${exp_name}.job"
    curdate=$(date +'%Y_%m_%d__%H_%M_%N')

    echo "#!/bin/bash
#SBATCH --job-name=${exp_name}.job
#SBATCH --output=${out_directory}/${exp_name}.out
#SBATCH --error=${out_directory}/${exp_name}.err
#SBATCH --time=1-00:00
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
