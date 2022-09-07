import argparse
from multiprocessing.sharedctypes import Value
from circuitmap import NeuralDemixer
from pc_subtractr_network import Subtractr
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

sys.path.append('../')
import grid_utils as util
import subtract_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('--subtractr_checkpoint', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--demixer_checkpoint', type=str,
        default='../../circuit_mapping/demixers/nwd_ee_ChroME1.ckpt')
    parser.add_argument('--dataset_path', type=str,
        default='../data/masato/B6WT_AAV_hsyn_chrome2f_gcamp8/preprocessed/220308_B6_Chrome2fGC8_030822_Cell2_opsPositive_A_planes_cmReformat.mat')

    # option to run demixer on raw traces before subtraction
    parser.add_argument('--show_raw_demixed', action='store_true')
    parser.add_argument('--no_show_raw_demixed', action='store_false', dest='show_raw_demixed')
    parser.set_defaults(show_raw_demixed=False)

    # option to run Lasso for multispot data
    parser.add_argument('--use_lasso', action='store_true')
    parser.add_argument('--no_use_lasso', action='store_false', dest='use_lasso')
    parser.set_defaults(use_lasso=False)

    parser.add_argument('--network_type', type=str, default='demixer')
    args = parser.parse_args()

    if args.network_type == 'demixer':
        # import pdb; pdb.set_trace()
        net = NeuralDemixer(path=args.subtractr_checkpoint,
        unet_args=dict(
            down_filter_sizes=(16, 32, 64, 128),
            up_filter_sizes=(64, 32, 16, 4),
        ))
    elif args.network_type == 'subtractr':
        net = Subtractr().load_from_checkpoint(args.subtractr_checkpoint)
    else:
        raise ValueError('unrecognized network type: %s' % args.network_type)

    with h5py.File(args.dataset_path) as f:
        pscs = np.array(f['pscs']).T
        stim_mat = np.array(f['stimulus_matrix']).T
        targets = np.array(f['targets']).T
        powers = np.max(stim_mat, axis=0)

        if 'num_pulses_per_holo' in f:
            npulses = np.array(f['num_pulses_per_holo'], dtype=int).item()
        else:
            npulses = 1

    stim_mat_list, pscs_list, powers_list = util.separate_by_pulses(stim_mat, pscs, npulses=npulses)
    stim_mat = stim_mat_list[-1]
    pscs = pscs_list[-1]
    powers = powers_list[-1]

    # get rid of any trials where we didn't actually stim
    good_idxs = (powers > 0)
    pscs = pscs[good_idxs,:]
    stim_mat = stim_mat[:,good_idxs]
    powers = powers[good_idxs]

    results = subtract_utils.run_network_subtraction_pipeline(
        pscs, powers, targets, stim_mat,
        args.demixer_checkpoint, net,
        run_raw_demix=args.show_raw_demixed,
    )

    # For multispot data, run lasso on raw, subtracted, and demixed
    # instead of using the maps from the results dict
    if args.use_lasso:
        raw_lasso_resp = util.circuitmap_lasso_cv(stim_mat, pscs)[0]
        subtracted_lasso_resp = util.circuitmap_lasso_cv(stim_mat, results['subtracted_matrix'])[0]
        demixed_lasso_resp = util.circuitmap_lasso_cv(stim_mat, results['demixed_matrix'])[0]

        results['raw_map'] = raw_lasso_resp.reshape(*(results['raw_map'].shape))
        results['subtracted_map'] = subtracted_lasso_resp.reshape(*(results['subtracted_map'].shape))
        results['demixed_map'] = demixed_lasso_resp.reshape(*(results['demixed_map'].shape))
    
    # make map figure
    num_planes = results['raw_map'].shape[-1]
    fig1 = plt.figure(figsize=(4 * 3, num_planes), dpi=300, facecolor='white')
    util.plot_multi_means(fig1,
        [
            results['raw_map'],
            results['subtracted_map'],
            results['demixed_map'],
        ], 
        np.arange(num_planes),
        map_names=['raw', 'subtr.', 'demixed'],
        cmaps=['magma', 'magma', 'magma', 'magma'],
        # cbar_labels=['EPSQ (nC)'],
        # zlabels=['subtr', 'demix'],
        # map_names=['raw', 'subtr', 'demix'],
        vranges=[(0,15), (0,15), (0,15), (0,15)],
        powers=np.unique(powers),
        show_powers=(True, True, True, True)
    )
    dset_name = os.path.basename(args.dataset_path)
    maps_outpath = os.path.join(args.save_path,
        args.experiment_name + '_' + dset_name + '_maps.png')
    plt.savefig(maps_outpath)

    # make traces figure
    fig2, axs = subtract_utils.plot_subtraction_comparison(
        results['raw_tensor'],
        [results['est_tensor']], # est
        [results['subtracted_tensor'],], # subtracted
        [results['demixed_tensor'],], # demixed
        np.unique(powers),
        num_plots_per_power=4,
    )
    traces_outpath = os.path.join(args.save_path,
        args.experiment_name + '_' + dset_name + '_traces.png')
    plt.savefig(traces_outpath)
