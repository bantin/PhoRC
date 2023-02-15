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
    parser.add_argument('--xla-allocator-platform',
                        action='store_true', default=False)
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


def split_results_dict(results):
    """Split the results dict into two dicts, one for single target and one for multi-target trials.

    Parameters
    ----------
    results : dict
        Dictionary of results from running the preprocessing pipeline.
        Expected to contain the keys: stim_mat, pscs, demixed, subtracted, est, raw, raw_demixed
    
    Returns
    -------
    results : dict
        Dictionary of results from running the preprocessing pipeline.
        Contains the keys: stim_mat_single, pscs_single, demixed_single,
        subtracted_single, est_single, raw_single, raw_demixed_single and
        stim_mat_multi, pscs_multi, demixed_multi, subtracted_multi, est_multi,
        raw_multi, raw_demixed_multi
    """
    stim_mat, raw, demixed, subtracted, est, raw_demixed = results['stim_mat'], results['raw'], results[
        'demixed'], results['subtracted'], results['est'], results['raw_demixed']
    targets = results['targets']
    unique_target_counts = np.unique(np.sum(stim_mat > 0, axis=0))
    max_target_count = unique_target_counts[-1]
    min_target_count = unique_target_counts[0]
    
    results_new = {}

    # fill in singlespot results
    if min_target_count == 1:
        results_new['singlespot'] = {}
        these_idxs = np.sum(stim_mat > 0, axis=0) == min_target_count
        results_new['singlespot']['stim_mat'] = stim_mat[:, these_idxs]
        results_new['singlespot']['demixed'] = demixed[these_idxs]
        results_new['singlespot']['subtracted'] = subtracted[these_idxs]
        results_new['singlespot']['est'] = est[these_idxs]
        results_new['singlespot']['raw'] = raw[these_idxs]
        results_new['singlespot']['raw_demixed'] = raw_demixed[these_idxs]

    # fill in multispot results if available
    if max_target_count > 1:
        results_new['multispot'] = {}
        these_idxs = np.sum(stim_mat > 0, axis=0) == max_target_count
        results_new['multispot']['stim_mat'] = stim_mat[:, these_idxs]
        results_new['multispot']['demixed'] = demixed[these_idxs]
        results_new['multispot']['subtracted'] = subtracted[these_idxs]
        results_new['multispot']['est'] = est[these_idxs]
        results_new['multispot']['raw'] = raw[these_idxs]
        results_new['multispot']['raw_demixed'] = raw_demixed[these_idxs]

    results_new['targets'] = targets
    return results_new



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
    pscs = pscs[good_idxs, :]
    stim_mat = stim_mat[:, good_idxs]
    powers = powers[good_idxs]

    # Run the preprocessing pipeline creating demixed data both
    # with and without the subtraction step.
    results = utils.run_preprocessing_pipeline(
        pscs, powers, targets, stim_mat, args.demixer_path,
        subtractr_path=args.subtractr_path,
        subtract_pc=True,
        run_raw_demixed=True,
        rank=args.rank,
        constrain_V=args.constrain_V,
        baseline=args.baseline,
        subtract_baseline=args.subtract_baseline,
        batch_size=args.batch_size,
        extended_baseline=args.extended_baseline,
    )

    # split the results dict into single and multi target count groups
    results = split_results_dict(results)
    for target_count_label in ['singlespot', 'multispot']:
        if target_count_label not in results:
            continue
        for psc_label, output_label in zip(['demixed', 'raw_demixed'], ['subtraction_on', 'subtraction_off']):

            print('Running caviar with %s' % output_label)
            pscs_curr = results[target_count_label][psc_label]
            stim_mat_curr = results[target_count_label]['stim_mat']
            N = stim_mat_curr.shape[0]
            model = cm.Model(N)
            model.fit(pscs_curr,
                stim_mat_curr,
                method='caviar',
                fit_options={
                    'msrmp': args.msrmp,
                    'iters': args.iters,
                    'save_histories': args.save_histories,
                    'minimum_spike_count': args.minimum_spike_count}
            )
            results[target_count_label]['model_state_' + output_label] = model.state



    # save args used to get the result
    argstr = json.dumps(args.__dict__)
    outpath = dset_name + '_' + args.file_suffix + '_results.npz'
    np.savez_compressed(outpath, **results, args=argstr)
