#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --time=00:22:00         
#SBATCH --job-name=multi_round

### %J ist der JobName
#SBATCH --output=/home/fu494742/MasterArbeit/slum_outputs/output_MR_%j.txt    
#SBATCH --array=1-80

### Program Code
cd /home/fu494742/MasterArbeit/code

### Load Virtual Python
source /home/fu494742/MasterArbeit/.venv/bin/activate 

# config_path="/home/fu494742/MasterArbeit/code/configs/md_mr_mp_MR/ML_Z.yaml" # num_shots: 1k # 19:00 min
config_path="/home/fu494742/MasterArbeit/code/configs/md_mr_mp_MR/ML_X.yaml"  # 80 started
# config_path="/home/fu494742/MasterArbeit/code/configs/md_mr_mp_MR/MWPM_Z.yaml" # 10 started num_shots: 10k # 6:20 
# config_path="/home/fu494742/MasterArbeit/code/configs/md_mr_mp_MR/MWPM_X.yaml" # 10 started

output_folder="md_1r_mp_MR"

python slurm_wrapper.py -c $config_path -o $output_folder #-u ${SLURM_ARRAY_TASK_ID}