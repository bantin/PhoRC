#!/bin/bash
#
#
#SBATCH --job-name=circuitmap_grids
#SBATCH -c 8                   
#SBATCH --time=10:00:00             
#SBATCH --mem-per-cpu=4GB       
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

echo "Denoising m.s. grids"

source ~/.bashrc
ml cuda/11.2.0 cudnn/8.2.1.32
conda activate subtraction

python run_isi_sweep.py "$@"


# End of script
