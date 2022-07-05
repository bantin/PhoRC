#!/bin/bash
#
#
#SBATCH --job-name=ai203_grids
#SBATCH -c 8                   
#SBATCH --time=10:00:00             
#SBATCH --mem-per-cpu=4GB       
#SBATCH --gres=gpu:a40:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --array=1
 
echo "Denoising m.s. grids"

source ~/.bashrc
conda activate circuitmap
ml cuda/11.2.0 cudnn/8.2.1.32

arg_str="$*"

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p /share/ctn/users/ba2617/mbcs_grids/filelist.txt )

python run_gridmap.py --iters 50 --dataset-path $LINE --demixer-checkpoint "/home/ba2617/circuit_mapping/demixers/nwd_ee_ChroME1.ckpt"

# End of script
