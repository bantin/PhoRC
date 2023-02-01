import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from circuitmap import NeuralDemixer
import circuitmap as cm
import h5py


import sys
import json

import subtractr
import subtractr.utils as utils
import os
import argparse

plt.rcParams['figure.dpi'] = 300



def parse_fit_options(argseq):
    parser = argparse.ArgumentParser(
        description='Opsin subtraction + CAVIaR for Grid Denoising')
    
    # caviar args
    parser = utils.add_caviar_args(parser=parser)

    # subtraction args
    parser = utils.add_subtraction_args(parser=parser)

    # run args
    parser.add_argument('--xla-allocator-platform', action='store_true', default=False)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--demixer_path', type=str,
        default="/Users/Bantin/Documents/Columbia/Projects/2p-opto/circuit_mapping/demixers/nwd_ee_ChroME1.ckpt")

    # optionally add a suffix to the saved filename, e.g to specify what arguments were used
    parser.add_argument('--file_suffix', type=str, default="")

    # whether we're working with grid or targeted data    
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--targeted', dest='grid', action='store_false')
    parser.set_defaults(grid=True)

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
    
    # Run (optional) subtraction and demixing
    results = utils.run_preprocessing_pipeline(pscs, powers, targets, stim_mat, args.demixer_path, 
        subtract_pc=args.subtract_pc, 
        subtractr_path=args.subtractr_path, 
        stim_start=args.stim_start_idx, stim_end=args.stim_end_idx,
        rank=args.rank, constrain_V=args.constrain_V, baseline=args.baseline,
        subtract_baseline=args.subtract_baseline,
        batch_size=args.batch_size,)
 
    if args.grid:
        print('grid-data flag is set. Adding tensors and maps to results dict for plotting.')
        results = utils.add_grid_results(results)
    else:
        print('Running on targeted data. Skipping grid tensors and maps...')

    # check whether the dataset has both single and multi target,
    # and if so run CAVIaR separately on each.
    unique_target_counts = np.unique(np.sum(results['stim_mat'] > 0, axis=0))
    if args.run_caviar:
        if len(unique_target_counts) > 1:
            print('Running CAVIaR separately on each target count...')
            # there is an issue here where some datasets have more than 1 spot or 10 spot maps,
            # only take the minimal or maximal number of spots present.
            assert unique_target_counts[0] == 1, "First unique target count is not 1."
            target_count_single = 1
            target_count_multi = unique_target_counts[-1] # np.unique returns sorted values
            for target_count, label in zip(
                    [target_count_single, target_count_multi],
                    ['single', 'multi']):
                print('Running CAVIaR on {} target trials...'.format(target_count))
                these_idxs = np.sum(results['stim_mat'] > 0, axis=0) == target_count
                stim_mat_curr = results['stim_mat'][:,these_idxs]
                psc_curr = results['demixed'][these_idxs]
                N = stim_mat.shape[0]
                model = cm.Model(N)
                model.fit(
                    psc_curr,
                    stim_mat_curr,
                    method='caviar',
                    fit_options={
                        'msrmp': args.msrmp,
                        'iters': args.iters,
                        'save_histories': args.save_histories,
                        'minimum_spike_count': args.minimum_spike_count}
                )
                results['model_state_' + label] = model.state
                results['raw_' + label] = results['raw'][these_idxs]
                results['est_' + label] = results['est'][these_idxs]
                results['subtracted_' + label] = results['subtracted'][these_idxs]
                results['demixed_' + label] = results['demixed'][these_idxs]
                results['stim_mat_' + label] = results['stim_mat'][:,these_idxs]

            # delete the original keys which are not separated by target count
            del results['raw']
            del results['est']
            del results['subtracted']
            del results['demixed']
            del results['stim_mat']

        else:
            print('Found only %d spot data' % unique_target_counts[0])
            if unique_target_counts[0] == 1:
                label = 'single'
            else:
                label = 'multi'
            print('Running CAVIaR on all trials (%s)...' % label)
            N = stim_mat.shape[0]
            model = cm.Model(N)
            model.fit(results['demixed'],
                stim_mat,
                method='caviar',
                fit_options={
                    'msrmp': args.msrmp,
                    'iters': args.iters,
                    'save_histories': args.save_histories,
                    'minimum_spike_count': args.minimum_spike_count}
            )

            results['model_state'] = model.state
    
    # save args used to get the result
    argstr = json.dumps(args.__dict__)
    outpath = dset_name + '_' + args.file_suffix + '_results.npz'
    np.savez_compressed(outpath, **results, args=argstr)


