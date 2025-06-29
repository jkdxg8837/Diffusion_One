#!/bin/bash
#
#SBATCH --job-name=diff_eval # Job name for tracking
#SBATCH --partition=mlsys-ada  # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32      # Number of CPU threads used by your job
#SBATCH --gres=gpu:2          # Number of GPUs to use 
# 80GB of system RAM will be allocated with 1 GPU specified

#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=../joboutput/eval_joboutput_%j.out # Standard out from your job
#SBATCH --error=../joboutput/eval_joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
source ~/.bashrc
conda activate /dcs/large/u5649209/envs/omini

## Execute your program(s) ##
python eval_dreambooth.py