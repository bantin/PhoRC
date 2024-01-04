#!/bin/bash
#
#
#SBATCH --job-name=batch_size_sweep
#SBATCH -c 8                   
#SBATCH --time=18:00:00             
#SBATCH --mem-per-cpu=8GB       
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

echo "Running CAVIaR before/after subtraction"

source ~/.bashrc
conda activate subtraction
ml cuda/11.2.0 cudnn/8.2.1.32

python batch_size_sweep.py "$@"


# End of script
