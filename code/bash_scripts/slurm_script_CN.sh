#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --time=00:05:00         
#SBATCH --job-name=circ_noise

### %J ist der JobName
#SBATCH --output=/home/fu494742/MasterArbeit/slum_outputs/output_CN_%j.txt    
#SBATCH --array=1-48

### Program Code
cd /home/fu494742/MasterArbeit/code

### Load Virtual Python
source /home/fu494742/MasterArbeit/.venv/bin/activate 

config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_CN/ML_Z.yaml" # 3:40 min # num shots 1k 
# config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_CN/ML_X.yaml" # 4:00 min # num_shots 1k 
# config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_CN/MWPM_Z.yaml" # 2:00 min # num_shots 10k 
# config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_CN/MWPM_X.yaml" # 1:40min # num_shots 10k  

output_folder="md_1r_mp_CN"

python slurm_wrapper.py -c $config_path -o $output_folder -u ${SLURM_ARRAY_TASK_ID}