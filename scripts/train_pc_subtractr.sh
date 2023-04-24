#!/bin/bash
#
#
#SBATCH --job-name=train_pc_subtractr
#SBATCH -c 8                   
#SBATCH --time=3-00:00:00             
#SBATCH --mem-per-cpu=4GB
#SBATCH --gres=gpu:a40:1

#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

echo "Denoising m.s. grids"

source ~/.bashrc
conda activate subtraction
ml cuda/11.2.0 cudnn/8.2.1.32

python pc_subtractr_network.py --num_train 500000 --num_test 1000 --use_onecyclelr --onecyclelr_max_lr 0.1 --max_epochs 100 --model_type MultiTraceConv --num_traces_per_expt 100 --batch_size 8 --data_on_disk --data_save_path /local/ --target_gp_scale 0.01 --min_pc_fraction 0.7 --max_pc_fraction 1.0 --gpus 1 --tau_o_min 2 --tau_o_max 15 --onset_latency_ms 0.0 --inhibitory_pscs


# End of script
