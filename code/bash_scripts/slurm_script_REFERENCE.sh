#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --time=01:30:00         
#SBATCH --job-name=reference

### %J ist der JobName
#SBATCH --output=/home/fu494742/MasterArbeit/slum_outputs/output_REFERENCE_%j.txt    
#SBATCH --array=1-40

### Program Code
cd /home/fu494742/MasterArbeit/code

### Load Virtual Python
source /home/fu494742/MasterArbeit/.venv/bin/activate 

# d [3,5,7,9] (0:32:00 last time) need 40 sets each (2.5k shots)
# output_folder="md_mr_mp_reference_v2"
# config_path="/home/fu494742/MasterArbeit/code/configs/md_mr_mp_reference/MWPM_Z.yaml"
# config_path="/home/fu494742/MasterArbeit/code/configs/md_mr_mp_reference/MWPM_X.yaml"

python slurm_wrapper.py -c $config_path -o $output_folder -u ${SLURM_ARRAY_TASK_ID}