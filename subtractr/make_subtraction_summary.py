import argparse
from multiprocessing.sharedctypes import Value
from circuitmap import NeuralDemixer
from pc_subtractr_network import Subtractr
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import os

import subtractr.utils as utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('--subtractr_checkpoint', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--demixer_checkpoint', type=str,
        default='../../circuit_mapping/demixers/nwd_ee_ChroME1.ckpt')
    parser.add_argument('--dataset_path', type=str,
        default='../data/masato/B6WT_AAV_hsyn_chrome2f_gcamp8/preprocessed/220308_B6_Chrome2fGC8_030822_Cell2_opsPositive_A_planes_cmReformat.mat')
    parser.add_argument('--subtraction_method', type=str, default='network')

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

    dset_name = os.path.basename(args.dataset_path)
    if args.subtraction_method == 'network':
        assert args.subtractr_checkpoint, "Must pass a subtractr checkpoint."
        if args.network_type == 'demixer':
            net = NeuralDemixer(path=args.subtractr_checkpoint,
            unet_args=dict(
                down_filter_sizes=(16, 32, 64, 128),
                up_filter_sizes=(64, 32, 16, 4),
            ))
        elif args.network_type == 'subtractr':
            net = Subtractr.load_from_checkpoint(args.subtractr_checkpoint)
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

    stim_mat_list, pscs_list, powers_list = utils.separate_by_pulses(stim_mat, pscs, npulses=npulses)
    stim_mat = stim_mat_list[-1]
    pscs = pscs_list[-1]
    powers = powers_list[-1]

    # get rid of any trials where we didn't actually stim
    good_idxs = (powers > 0)
    pscs = pscs[good_idxs,:]
    stim_mat = stim_mat[:,good_idxs]
    powers = powers[good_idxs]

    if args.subtraction_method == 'network':
        results = utils.run_network_subtraction_pipeline(
            pscs, powers, targets, stim_mat,
            args.demixer_checkpoint, net,
            run_raw_demix=args.show_raw_demixed,
        )
    elif args.subtraction_method == 'lowrank':
        results = utils.run_subtraction_pipeline(
            pscs, powers, targets, stim_mat, 
            args.demixer_checkpoint,
            batch_size=100,
        )

    # add maps and tensors for plotting
    results = utils.add_grid_results(results)

    # sort results so that reshaping caviar weights works
    # results = util.sort_results(results)
    # TODO: add this back for multispot data

    # For multispot data, run lasso on raw, subtracted, and demixed
    # instead of using the maps from the results dict

    if np.sum(stim_mat[:,0] > 0) > 1: # check if multispot data
        raw_lasso_resp = utils.circuitmap_lasso_cv(stim_mat, results['raw'])[0]
        subtracted_lasso_resp = utils.circuitmap_lasso_cv(stim_mat, results['subtracted'])[0]
        demixed_lasso_resp = utils.circuitmap_lasso_cv(stim_mat, results['demixed'])[0]

        grid_dims = results['raw_map'].shape[1:]
        results['raw_map'] = utils.reshape_lasso_response(
            raw_lasso_resp, results['targets'], grid_dims)
        results['subtracted_map'] = utils.reshape_lasso_response(
            subtracted_lasso_resp, results['targets'], grid_dims)
        results['demixed_map'] = utils.reshape_lasso_response(
            demixed_lasso_resp, results['targets'], grid_dims)
    
    # make map figure
    maps_to_plot = [results['raw_map'], results['subtracted_map'], results['demixed_map']]
    map_names=['raw', 'subtr.', 'demixed']

    # check whether we have run caviar on the dataset, if so, add the inferred weights
    # in a 4th panel 
    if 'model_state' in results:

        # caviar_map = results['model_state']['mu'].reshape(1, *results['raw_map'].shape[1:])
        grid_dims = results['raw_map'].shape[1:]
        caviar_weights = results['model_state']['mu']
        caviar_map = utils.reshape_lasso_response(
            caviar_weights[None, :], results['targets'], grid_dims,
        )
        maps_to_plot.append(caviar_map)
        map_names.append('caviar')

    num_planes = results['raw_map'].shape[-1]
    fig1 = plt.figure(figsize=(4 * 3, num_planes), dpi=300, facecolor='white')
    utils.plot_multi_means(fig1,
        maps_to_plot, 
        np.arange(num_planes),
        map_names=map_names,
        cmaps=['magma', 'magma', 'magma', 'magma'],
        # cbar_labels=['EPSQ (nC)'],
        # zlabels=['subtr', 'demix'],
        # map_names=['raw', 'subtr', 'demix'],
        # vranges=[(0,15), (0,15), (0,15), (0,15)],
        powers=np.unique(results['powers']),
        show_powers=(True, True, True, True)
    )
    plt.tight_layout()

    # make traces figure
    fig2 = utils.plot_subtraction_by_power(
        results['raw'],
        results['est'],
        results['subtracted'],
        results['demixed'],
        results['powers'],
    )

    if args.experiment_name is None:
        experiment_name = args.subtraction_method
    else:
        experiment_name = args.experiment_name
        
    outpath = os.path.join(args.save_path,
        experiment_name + '_' + dset_name + '_summary.pdf')
    print('saving figure to: %s' % outpath)
    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath)
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.close()
