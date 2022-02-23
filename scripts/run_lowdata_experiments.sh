
# We assume running this from the script directory
job_directory=$PWD/.job
out_directory=$PWD/.out
mkdir -p ${job_directory}
mkdir -p ${out_directory}

exp_names=(
    # 'MHUSHARE_SPLIT_s120'
    # 'MHUSHARE_SPLIT_s130'
    # 'MHUSHARE_SPLIT_s140'
    # 'SPLIT_ALL_s120'
    # 'SPLIT_ALL_s130'
    # 'SPLIT_ALL_s140'
    # 'MHUSHARE_ALL_s110'
    # 'MHUSHARE_ALL_s120'
    # 'MHUSHARE_ALL_s130'
    # 'MHUSHARE_ALL_s140'
    # 'TRUESHARE_SPLIT_s110'
    # 'TRUESHARE_SPLIT_s120'
    # 'TRUESHARE_SPLIT_s130'
    # 'TRUESHARE_SPLIT_s140'
    'TRUESHARE_SPLIT_s150'
    # 'TRUESHARE_SPLIT_s160'
    # 'TRUESHARE_SPLIT_s170'
    'SPLIT_ALL_s150'
    # 'SPLIT_ALL_s160'
    # 'SPLIT_ALL_s170'
    'MHUSHARE_ALL_s150'
    # 'MHUSHARE_ALL_s160'
    # 'MHUSHARE_ALL_s170'
    # 'MHUSHARE_SPLIT_s150'
    # 'MHUSHARE_SPLIT_s160'
    # 'MHUSHARE_SPLIT_s170'
    # 'TRUESHARE_ALL_s110'
    # 'TRUESHARE_ALL_s120'
    # 'TRUESHARE_ALL_s130'
    # 'TRUESHARE_ALL_s140'
    # 'TRUESHARE_ALL_s150'
    # 'STL_SPLIT_ankle_elbow_s0'
    'STL_SPLIT_shoulder_s1'
    # 'STL_SPLIT_wrist_s0'
    # 'STL_SPLIT_shoulder_s2'
    # 'STL_SPLIT_shoulder_s3'
    # 'STL_SPLIT_shoulder_s4'
    # 'STL_SPLIT_shoulder_s5'
    'STL_SPLIT_ankle_elbow_s1'
    )

cmds=(
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare split split --epochs 150 --experimentname MHUSHARE_SPLIT --device cuda:0 --scarcities 2 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare split split --epochs 150 --experimentname MHUSHARE_SPLIT --device cuda:0 --scarcities 3 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare split split --epochs 150 --experimentname MHUSHARE_SPLIT --device cuda:0 --scarcities 4 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures split split split split split split split split --epochs 150 --experimentname SPLIT_ALL --device cuda:0 --scarcities 2 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures split split split split split split split split --epochs 150 --experimentname SPLIT_ALL --device cuda:0 --scarcities 3 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures split split split split split split split split --epochs 150 --experimentname SPLIT_ALL --device cuda:0 --scarcities 4 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare --epochs 150 --experimentname MHUSHARE_ALL --device cuda:0 --scarcities 1 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare --epochs 150 --experimentname MHUSHARE_ALL --device cuda:0 --scarcities 2 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare --epochs 150 --experimentname MHUSHARE_ALL --device cuda:0 --scarcities 3 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare --epochs 150 --experimentname MHUSHARE_ALL --device cuda:0 --scarcities 4 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare split split --epochs 150 --experimentname TRUESHARE_SPLIT --device cuda:0 --scarcities 1 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare split split --epochs 150 --experimentname TRUESHARE_SPLIT --device cuda:0 --scarcities 2 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare split split --epochs 150 --experimentname TRUESHARE_SPLIT --device cuda:0 --scarcities 3 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare split split --epochs 150 --experimentname TRUESHARE_SPLIT --device cuda:0 --scarcities 4 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare split split --epochs 150 --experimentname TRUESHARE_SPLIT --device cuda:0 --scarcities 5 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare split split --epochs 150 --experimentname TRUESHARE_SPLIT --device cuda:0 --scarcities 6 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare split split --epochs 150 --experimentname TRUESHARE_SPLIT --device cuda:0 --scarcities 7 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures split split split split split split split split --epochs 150 --experimentname SPLIT_ALL --device cuda:0 --scarcities 5 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures split split split split split split split split --epochs 150 --experimentname SPLIT_ALL --device cuda:0 --scarcities 6 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures split split split split split split split split --epochs 150 --experimentname SPLIT_ALL --device cuda:0 --scarcities 7 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare --epochs 150 --experimentname MHUSHARE_ALL --device cuda:0 --scarcities 5 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare --epochs 150 --experimentname MHUSHARE_ALL --device cuda:0 --scarcities 6 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare --epochs 150 --experimentname MHUSHARE_ALL --device cuda:0 --scarcities 7 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare split split --epochs 150 --experimentname MHUSHARE_SPLIT --device cuda:0 --scarcities 5 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare split split --epochs 150 --experimentname MHUSHARE_SPLIT --device cuda:0 --scarcities 6 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures mhushare mhushare mhushare mhushare mhushare mhushare split split --epochs 150 --experimentname MHUSHARE_SPLIT --device cuda:0 --scarcities 7 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare trueshare trueshare --epochs 150 --experimentname TRUESHARE_ALL --device cuda:0 --scarcities 1 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare trueshare trueshare --epochs 150 --experimentname TRUESHARE_ALL --device cuda:0 --scarcities 2 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare trueshare trueshare --epochs 150 --experimentname TRUESHARE_ALL --device cuda:0 --scarcities 3 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare trueshare trueshare --epochs 150 --experimentname TRUESHARE_ALL --device cuda:0 --scarcities 4 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow shoulder wrist --weighting naive --blockstructures trueshare trueshare trueshare trueshare trueshare trueshare trueshare trueshare --epochs 150 --experimentname TRUESHARE_ALL --device cuda:0 --scarcities 5 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets ankle_elbow --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname STL_SPLIT --device cuda:0 --scarcities 0 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname STL_SPLIT --device cuda:0 --scarcities 1 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets wrist --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname STL_SPLIT --device cuda:0 --scarcities 0 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname STL_SPLIT --device cuda:0 --scarcities 2 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname STL_SPLIT --device cuda:0 --scarcities 3 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname STL_SPLIT --device cuda:0 --scarcities 4 --savefreq 1 --numworkers 1'
    # 'python mtl.py --datasets shoulder --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname STL_SPLIT --device cuda:0 --scarcities 5 --savefreq 1 --numworkers 1'
    'python mtl.py --datasets ankle_elbow --weighting same --blockstructures split split split split split split split split --epochs 150 --experimentname STL_SPLIT --device cuda:0 --scarcities 1 --savefreq 1 --numworkers 1'
    )


for i in ${!exp_names[@]}; do
    
    exp_name="${exp_names[$i]}"

    job_file="${job_directory}/${exp_name}.job"
    curdate=$(date +'%Y_%m_%d__%H_%M_%N')

    echo "#!/bin/bash
#SBATCH --job-name=${exp_name}.job
#SBATCH --output=${out_directory}/${exp_name}.out
#SBATCH --error=${out_directory}/${exp_name}.err
#SBATCH --time=2-00:00
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
