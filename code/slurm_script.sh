#!/usr/bin/zsh 

### Job Parameters 

#SBATCH --time=00:05:00         # Run time of 15 minutes

#SBATCH --job-name=test_script
### %J ist der JobName
# redirects stdout and stderr to stdout.txt
#SBATCH --output=/home/fu494742/MasterArbeit/slum_outputs/output_%j.txt    

### Program Code
cd /home/fu494742/MasterArbeit/code

### Load Virtual Python
source /home/fu494742/MasterArbeit/.venv/bin/activate 

config_path="/home/fu494742/MasterArbeit/code/configs/md_1r_mp_BN/ML_X.yaml"
output_folder="md_1r_mp_BN"

python slurm_wrapper.py -c $config_path -o $output_folder  #-u $SLURM_JOB_ID