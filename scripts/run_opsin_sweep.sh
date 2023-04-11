#!/bin/bash
#
#
#SBATCH --job-name=subtractr_opsin_sweep
#SBATCH -c 8                   
#SBATCH --time=1-10:00:00             
#SBATCH --mem-per-cpu=4GB       
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

echo "Running opsin sweep for low-rank subtraction"

source ~/.bashrc
ml cuda/11.2.0 cudnn/8.2.1.32
conda activate subtraction

python run_opsin_sweep.py "$@"


# End of script
