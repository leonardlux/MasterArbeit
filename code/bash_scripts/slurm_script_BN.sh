#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --time=00:07:00         
#SBATCH --job-name=basic_noise

### %J ist der JobName
#SBATCH --output=/home/fu494742/MasterArbeit/slum_outputs/output_BN_%j.txt    
#SBATCH --array=1-100

### Program Code
cd /home/fu494742/MasterArbeit/code

### Load Virtual Pythoncode/configs/md_1r_mp_BN/slurm_script.sh
source /home/fu494742/MasterArbeit/.venv/bin/activate 

# all done in current configuration (15.06.2026)
# config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_BN/ML_Z.yaml" # 1k: 3:00 min # done
# config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_BN/ML_X.yaml" # 1k: 3:20 min # done 
# config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_BN/MWPM_Z.yaml" # 10k: (have 10)
# config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_BN/MWPM_X.yaml" # 10k: (have 10) 

output_folder="md_1r_mp_BN_v2_fault_det"

python slurm_wrapper.py -c $config_path -o $output_folder -u ${SLURM_ARRAY_TASK_ID}