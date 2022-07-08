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

python run_gridmap.py --msrmp 0.3 --iters 50 --dataset-path $1 --demixer-checkpoint "/home/ba2617/circuit_mapping/demixers/nwd_ee_ChroME1.ckpt"

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p /share/ctn/users/ba2617/mbcs_grids/filelist.txt )

# End of script
