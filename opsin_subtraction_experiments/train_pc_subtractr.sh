#!/bin/bash
#
#
#SBATCH --job-name=train_pc_subtractr
#SBATCH -c 8                   
#SBATCH --time=10:00:00             
#SBATCH --mem-per-cpu=4GB       
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

echo "Denoising m.s. grids"

source ~/.bashrc
conda activate circuitmap
ml cuda/11.2.0 cudnn/8.2.1.32

arg_str="$*"
python pc_subtractr_network.py $arg_str


# End of script
