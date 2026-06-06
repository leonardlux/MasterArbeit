#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --time=00:10:00         
#SBATCH --job-name=reference

### %J ist der JobName
#SBATCH --output=/home/fu494742/MasterArbeit/slum_outputs/output_REFERENCE_%j.txt    
#SBATCH --array=1-40

### Program Code
cd /home/fu494742/MasterArbeit/code

### Load Virtual Python
source /home/fu494742/MasterArbeit/.venv/bin/activate 

# 2k shots each
config_path="/home/fu494742/MasterArbeit/code/configs/md_mr_mp_reference_short/MWPM_X.yaml"
config_path="/home/fu494742/MasterArbeit/code/configs/md_mr_mp_reference_short/MWPM_Z.yaml"

output_folder="md_mr_mp_reference_short"

python slurm_wrapper.py -c $config_path -o $output_folder -u ${SLURM_ARRAY_TASK_ID}