#!/bin/bash
#
#
#SBATCH --job-name=circuitmap_grids
#SBATCH -c 8                   
#SBATCH --time=10:00:00             
#SBATCH --mem-per-cpu=4GB       
#SBATCH --gres=gpu:a40:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

echo "Denoising m.s. grids"

source ~/.bashrc
conda activate circuitmap
ml cuda/11.2.0 cudnn/8.2.1.32

stat $1
stat $2
python run_gridmap.py --msrmp 0.3 --iters 50 --dataset-path $1 --demixer-checkpoint $2 --run-caviar --subtract-pc


# End of script
