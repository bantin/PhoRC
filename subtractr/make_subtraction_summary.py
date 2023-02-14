import argparse
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
    parser.add_argument('--save_path', type=str, default='.')
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--demixer_path', type=str,
        default='../../circuit_mapping/demixers/nwd_ee_ChroME1.ckpt')
    parser.add_argument('--dataset_path', type=str,
        default='../data/masato/B6WT_AAV_hsyn_chrome2f_gcamp8/preprocessed/220308_B6_Chrome2fGC8_030822_Cell2_opsPositive_A_planes_cmReformat.mat')
    
    # Add args for subtraction method
    parser = utils.add_subtraction_args(parser=parser)

    # whether we're working with grid or targeted data    
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--targeted', dest='grid', action='store_false')
    parser.set_defaults(grid=True)

    # option to run demixer on raw traces before subtraction
    parser.add_argument('--show_raw_demixed', action='store_true')
    parser.add_argument('--no_show_raw_demixed', action='store_false', dest='show_raw_demixed')
    parser.set_defaults(show_raw_demixed=False)

    # option to run Lasso for multispot data
    parser.add_argument('--use_lasso', action='store_true')
    parser.add_argument('--no_use_lasso', action='store_false', dest='use_lasso')
    parser.set_defaults(use_lasso=False)

    # add option for extended baseline
    parser.add_argument('--extended_baseline', action='store_true')
    parser.add_argument('--no_extended_baseline', action='store_false', dest='extended_baseline')
    parser.set_defaults(extended_baseline=False)

    args = parser.parse_args()

    print(args)
    dset_name = os.path.basename(args.dataset_path)

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

    results = utils.run_preprocessing_pipeline(pscs, powers, targets, stim_mat, args.demixer_path, 
        subtract_pc=args.subtract_pc, 
        subtractr_path=args.subtractr_path, 
        stim_start=args.stim_start_idx, stim_end=args.stim_end_idx,
        rank=args.rank, constrain_V=args.constrain_V, baseline=args.baseline,
        subtract_baseline=args.subtract_baseline,
        batch_size=args.batch_size,
        extended_baseline=args.extended_baseline,)

    # If we're working with grid data, add maps and tensors for plotting,
    # then run Lasso on raw, subtracted, and demixed (for multispot), 
    # then reshape the Lasso responses into maps, then make map figure.
    # For targeted data, skip this.
    if args.grid:
        # add maps and tensors for plotting grid data
        results = utils.add_grid_results(results)


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
            vranges=[(0,15), (0,15), (0,15), (0,15)],
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
    if args.grid:
        pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.close()
