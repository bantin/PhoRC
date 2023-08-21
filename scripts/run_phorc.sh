#!/bin/bash
#
#
#SBATCH --job-name=subtract_photocurrents
#SBATCH -c 8                   
#SBATCH --time=10:00:00             
#SBATCH --mem-per-cpu=4GB       
#SBATCH --exclude=ax[11]

echo "Running PhoRC with low-rank subtraction"

source ~/.bashrc
ml cuda/11.2.0 cudnn/8.2.1.32
conda activate subtraction

python run_phorc.py "$@"


# End of script
