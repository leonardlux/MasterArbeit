#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --time=00:30:00         
#SBATCH --job-name=ml_test

### %J ist der JobName
#SBATCH --output=/home/fu494742/MasterArbeit/slum_outputs/output_MLtest_%j.txt    
#SBATCH --array=1-20

### Program Code
cd /home/fu494742/MasterArbeit/code

### Load Virtual Python
source /home/fu494742/MasterArbeit/.venv/bin/activate 

# config_path="/home/fu494742/MasterArbeit/code/configs/ml_test_aron_64/config.yaml" 
# config_path="/home/fu494742/MasterArbeit/code/configs/ml_test_basic_32/config.yaml" 
# config_path="/home/fu494742/MasterArbeit/code/configs/ml_test_basic_64/config.yaml" 
# config_path="/home/fu494742/MasterArbeit/code/configs/ml_test_log_32/config.yaml" 
config_path="/home/fu494742/MasterArbeit/code/configs/ml_test_log_64/config.yaml" 

# output_folder="ml_test_aron_64"
# output_folder="ml_test_basic_32"
# output_folder="ml_test_basic_64"
# output_folder="ml_test_log_32"
output_folder="ml_test_log_64"

python slurm_wrapper.py -c $config_path -o $output_folder -u ${SLURM_ARRAY_TASK_ID}