#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --time=21:15:00         
#SBATCH --job-name=circ_noise

### %J ist der JobName
#SBATCH --output=/home/fu494742/MasterArbeit/slum_outputs/output_CN_%j.txt    
#SBATCH --array=1-1

### Program Code
cd /home/fu494742/MasterArbeit/code

### Load Virtual Python
source /home/fu494742/MasterArbeit/.venv/bin/activate 

# normal (MWPM as FT check)
# config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_CN/ML_Z.yaml" # 3:40 min # num shots 1k 
# config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_CN/ML_X.yaml" # 4:00 min # num_shots 1k 
# config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_CN/MWPM_Z.yaml" # 2:00 min # num_shots 10k 
# config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_CN/MWPM_X.yaml" # 1:40min # num_shots 10k  
# output_folder="md_1r_mp_CN_v2_fault_det"

# ML as FT check
# config_path="/home/fu494742/MasterArbeit/code/configs/test_ft_ml/ML_Z.yaml" # 3:40 min # num shots 1k 
# config_path="/home/fu494742/MasterArbeit/code/configs/test_ft_ml/ML_X.yaml" # 4:00 min # num_shots 1k 
# config_path="/home/fu494742/MasterArbeit/code/configs/test_ft_ml/MWPM_Z.yaml" # 2:00 min # num_shots 10k 
# config_path="/home/fu494742/MasterArbeit/code/configs/test_ft_ml/MWPM_X.yaml" # 1:40min # num_shots 10k  
# output_folder="test_ft_ml"

# normal (MWPM) with small window up to distance 21 (10k shots = 21 hours)
config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_CN_small_window/ML_Z.yaml"
output_folder="many_d_small_p_1r"

python slurm_wrapper.py -c $config_path -o $output_folder -u ${SLURM_ARRAY_TASK_ID}