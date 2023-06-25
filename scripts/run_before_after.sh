#!/bin/bash
#
#
#SBATCH --job-name=circuitmap_grids
#SBATCH -c 8                   
#SBATCH --time=18:00:00             
#SBATCH --mem-per-cpu=8GB       
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ax[11,14]
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

echo "Running CAVIaR before/after subtraction"

source ~/.bashrc
conda activate subtraction
ml cuda/11.2.0 cudnn/8.2.1.32

python run_before_after_comparison.py "$@"


# End of script
