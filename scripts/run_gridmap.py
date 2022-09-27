import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from circuitmap import NeuralDemixer
import circuitmap as cm
import h5py


import sys
import json

sys.path.append('../')
import grid_utils as util
import subtract_utils as subtract_utils
import os
import argparse

plt.rcParams['figure.dpi'] = 300



def parse_fit_options(argseq):
    parser = argparse.ArgumentParser(
        description='Opsin subtraction + CAVIaR for Grid Denoising')
    
    # caviar args
    parser.add_argument('--run-caviar', action='store_true', default=False)
    parser.add_argument('--minimum-spike-count', type=int, default=3)
    parser.add_argument('--msrmp', type=float, default=0.3)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--save-histories', action='store_true', default=False)

    # run args
    parser.add_argument('--xla-allocator-platform', action='store_true', default=False)
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--demixer-checkpoint', type=str,
        default="/Users/Bantin/Documents/Columbia/Projects/2p-opto/circuit_mapping/demixers/nwd_ee_ChroME1.ckpt")
    parser.add_argument('--subtractr-checkpoint', type=str)

    # opsin subtract args
    parser.add_argument('--subtract-pc', action='store_true', default=False)
    parser.add_argument('--separate-by-power', action='store_true', default=False)
    parser.add_argument('--rank', type=int, default=1)

    args = parser.parse_args(argseq)

    if args.xla_allocator_platform:
        print('Setting XLA_PYTHON_CLIENT_ALLOCATOR to PLATFORM')
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    return args

if __name__ == "__main__":
    args = parse_fit_options(sys.argv[1:]) 
    dset_name = os.path.basename(args.dataset_path).split('.')[0]

    with h5py.File(args.dataset_path) as f:
        pscs = np.array(f['pscs']).T
        stim_mat = np.array(f['stimulus_matrix']).T
        targets = np.array(f['targets']).T
        powers = np.max(stim_mat, axis=0)

    # get rid of any trials where we didn't actually stim
    good_idxs = (powers > 0)
    pscs = pscs[good_idxs,:]
    stim_mat = stim_mat[:,good_idxs]
    powers = powers[good_idxs]
    

    # Optionally run photocurrent subtraction.
    # if no_op is True, the subtraction is a no_op and the following call
    # simply populates the results dictionary.
    no_op = (not args.subtract_pc)
    if not no_op:
        print('Running opsin subtraction pipeline...')
        subtractr_net = NeuralDemixer(path=args.subtractr_checkpoint,
            unet_args=dict(
                down_filter_sizes=(16, 32, 64, 128),
                up_filter_sizes=(64, 32, 16, 4),
            )
        )
    results = subtract_utils.run_network_subtraction_pipeline(pscs, powers, targets, stim_mat,
        args.demixer_checkpoint, subtractr_net)

    num_planes = results['raw_map'].shape[-1]

    # Run CAVIaR algorithm
    if args.run_caviar:
        N = stim_mat.shape[0]
        model = cm.Model(N)
        model.fit(results['demixed_matrix'],
            stim_mat,
            method='caviar',
            fit_options={
                'msrmp': args.msrmp,
                'iters': args.iters,
                'save_histories': args.save_histories,
                'minimum_spike_count': args.minimum_spike_count}
        )

        results['model_state'] = model.state
    results['powers'] = powers
    results['targets'] = targets
    results['stim_mat'] = stim_mat

    # save args used to get the result
    argstr = json.dumps(args.__dict__)
    outpath = dset_name + '_subtractr_caviar_results.npz'
    np.savez(outpath, **results, args=argstr)




