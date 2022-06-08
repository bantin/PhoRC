import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from circuitmap import NeuralDemixer
import circuitmap as cm


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
    parser.add_argument('--minimum-spike-count', type=int, default=3)
    parser.add_argument('--msrmp', type=float, default=0.3)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--save-histories', action='store_true', default=False)

    # run args
    parser.add_argument('--xla-allocator-platform', action='store_true', default=False)
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--demixer-checkpoint', type=str,
        default="/Users/Bantin/Documents/Columbia/Projects/2p-opto/circuit_mapping/demixers/nwd_ee_ChroME1.ckpt")

    # opsin subtract args
    parser.add_argument('--subtract-pc', action='store_true', default=True)
    parser.add_argument('--separate-by-power', action='store_true', default=False)
    parser.add_argument('--rank', type=int, default=1)

    args = parser.parse_args(argseq)

    if args.xla_allocator_platform:
        print('Setting XLA_PYTHON_CLIENT_ALLOCATOR to PLATFORM')
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    return args

if __name__ == "__main__":
    args = parse_fit_options(sys.argv[1:]) 
    dat = np.load(args.dataset_path, allow_pickle='True')

    pscs, I, L = dat['psc'], dat['I'], dat['L']
    dset_name = os.path.basename(args.dataset_path).split('.')[0]

    # Optionally run photocurrent subtraction.
    # if no_op is True, the subtraction is a no_op and the following call
    # simply populates the results dictionary.
    no_op = (not args.subtract_pc)
    results = subtract_utils.run_subtraction_pipeline(pscs, I, L,
        args.demixer_checkpoint, separate_by_power=False, rank=1, no_op=no_op)

    num_planes = results['raw_map'].shape[-1]

    # For multispot data, stim_mat will be in the data struct
    if 'stim_matrix' in dat:
        stim_mat = dat['stim_matrix']
    else:
        stim_mat = util.make_stim_matrix_singlespot(I, L)

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

    results['model'] = model
    results['I'] = I
    results['L'] = L
    results['stim_mat'] = stim_mat

    # save args used to get the result
    argstr = json.dumps(args.__dict__)
    outpath = dset_name + '_subtractr_caviar_results.npz'
    np.savez(outpath, **results, args=argstr)




