#!/bin/bash
#
#
#SBATCH --job-name=subtractr_validation
#SBATCH -c 8                   
#SBATCH --time=18:00:00             
#SBATCH --mem-per-cpu=4GB       
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

echo "Comparing subtractr network to low-rank model"

source ~/.bashrc
conda activate opsin_subtraction_experiments
ml cuda/11.2.0 cudnn/8.2.1.32

python run_synthetic_validation.py


# End of script
