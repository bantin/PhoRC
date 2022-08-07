#!/bin/bash
#
#
#SBATCH --job-name=ai203_grids
#SBATCH -c 8                   
#SBATCH --time=2:00:00             
#SBATCH --mem-per-cpu=4GB       
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --array=1
#SBATCH --exclude=ax[11-13]
 
echo "Denoising m.s. grids"

source ~/.bashrc
ml unload cuda
ml unload cudnn
conda activate jax_env

arg_str="$*"

LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p /share/ctn/users/ba2617/mbcs_grids/multispot/planes_filelist.txt )
echo $LINE

# $LINE should be a path to a dataset we want to process

python denoise_grids_singlespot_planewise.py $arg_str --dataset-path $LINE
# End of script
