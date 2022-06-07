#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys
import circuitmap as cm
from circuitmap import NeuralDemixer

import pickle
import glob
import neursim.utils as util
import os
from mpl_toolkits.axes_grid1 import ImageGrid
import argparse
import torch
import gc


def denoise_pscs_in_batches(psc, denoiser, batch_size=4096):

    num_pscs = psc.shape[0]
    num_batches = np.ceil(num_pscs / batch_size)
    den_psc_batched = [denoiser(batch, verbose=False)
                       for batch in np.array_split(psc, num_batches, axis=0)]
    return np.concatenate(den_psc_batched)


def make_grid_dataset(I, L):
    grid = util.make_grid_from_stim_locs(L)

    # Map from (x,y,z) -> index
    loc_map = {tuple(loc): idx for idx, loc in enumerate(
        grid.flat_grid)}

    # Map from index -> (x,y,z)
    idx_map = {idx: tuple(loc)
               for idx, loc in enumerate(grid.flat_grid)}

    num_neurons, num_trials = len(loc_map), len(I)
    stim = np.zeros((num_neurons, num_trials))

    # convert from L -> neuron_idx (number between 0 and num_neurons - 1)
    neuron_idxs = np.array([loc_map[tuple(loc)] for loc in L])

    # set entries of stim matrix. Since this is single spot data,
    # we can set the entire matrix at once (this is
    # trickier with multispot data)
    stim[neuron_idxs, np.arange(num_trials)] = I

    return stim, grid, loc_map, idx_map


def separate_data_by_plane(psc, den_psc, I, L):
    grid = util.make_grid_from_stim_locs(L)

    stim_mats = []
    pscs = []
    den_pscs = []
    Is = []
    Ls = []
    loc_maps = []
    idx_maps = []
    for z_idx, z in enumerate(grid.zs):

        this_plane = grid.flat_grid[:,-1] == z
        loc_map = {tuple(loc) : idx
            for idx, loc in enumerate(grid.flat_grid[this_plane])}
        idx_map = {idx: tuple(loc)
            for idx, loc in enumerate(grid.flat_grid)}

        # get number of neurons (i.e pixels) for a single plane
        # and number of trials in that plane.
        these_trials = L[:,-1] == z
        num_neurons, num_trials = len(loc_map), sum(these_trials)
        stim = np.zeros((num_neurons, num_trials))

        # convert from L -> neuron_idx (number between 0 and num_neurons - 1)
        neuron_idxs = np.array([loc_map[tuple(loc)] for loc in L[these_trials]])

        # set entries of stim matrix. Since this is single spot data,
        # we can set the entire matrix at once (this is
        # trickier with multispot data)
        stim[neuron_idxs, np.arange(num_trials)] = I[these_trials]

        # save stim and corresponding traces
        stim_mats.append(stim)
        pscs.append(psc[these_trials])
        den_pscs.append(den_psc[these_trials])
        Is.append(I[these_trials])
        Ls.append(L[these_trials])
        loc_maps.append(loc_map)
        idx_maps.append(idx_map)
    return stim_mats, pscs, den_pscs, Is, Ls, loc_maps, idx_maps


# def make_priors_mbcs(N, K):
#     beta_prior = 3e0 * np.ones(N)
#     mu_prior = np.zeros(N)
#     rate_prior = 1e-1 * np.ones(K)
#     shape_prior = np.ones(K)

#     priors = {
#         'beta': beta_prior,
#         'mu': mu_prior,
#         'shape': shape_prior,
#         'rate': rate_prior,
#     }
#     return priors


# def make_priors_vsns(N, K):
#     phi_prior = np.c_[0.125 * np.ones(N), 5 * np.ones(N)]
#     phi_cov_prior = np.array(
#         [np.array([[1e-1, 0], [0, 1e0]]) for _ in range(N)])
#     alpha_prior = 0.15 * np.ones(N)
#     beta_prior = 3e0 * np.ones(N)
#     mu_prior = np.zeros(N)
#     sigma_prior = np.ones(N)
#     sigma = 1

#     priors_vsns = {
#         'alpha': alpha_prior,
#         'beta': beta_prior,
#         'mu': mu_prior,
#         'phi': phi_prior,
#         'phi_cov': phi_cov_prior,
#         'shape': 1.,
#         'rate': sigma**2,
#     }

#     return priors_vsns


#def denoise_grid(model_type, fit_options, den_psc, stim):
#    all_models = []
#
#    if model_type == 'mbcs':
#        prior_fn = make_priors_mbcs
#        method = 'mbcs_spike_weighted_var_with_outliers'
#    elif model_type == 'variational_sns':
#        prior_fn = make_priors_vsns
#        method = 'cavi_sns'
#    else:
#        raise ValueError("invalid model type...")
#
#    N, K = stim.shape
#    priors = prior_fn(N, K)
#    model_params = {'N': N, 'model_type': model_type, 'priors': priors}
#    model = adaprobe.Model(**model_params)
#
#    # fit model and save
#    model.fit(psc, stim, fit_options=fit_options, method=method)
#
#    return model


def denoise_grid_planewise(minimax_spk_prob, iters, den_pscs, stims, trial_keep_prob=0.1):
    all_models = []

    for psc, stim in zip(den_pscs, stims):


        # subsample_trials
        num_trials = psc.shape[0]
        keep_idx = np.random.rand(num_trials) <= trial_keep_prob
        psc = psc[keep_idx,...]
        stim = stim[:, keep_idx]

        # create priors and model
        N, K = stim.shape
        model = cm.Model(N)

        # fit model and save
        model.fit(psc, stim, method='caviar',
            fit_options={'minimax_spk_prob':minimax_spk_prob, 'iters':iters})
        all_models.append(model)

    return all_models


def parse_fit_options(argseq):
    parser = argparse.ArgumentParser(
        description='CAVIaR for Grid Denoising')
    parser.add_argument('--minimax-spk-prob', type=float, default=0.3)
    parser.add_argument('--minimum-spike-count', type=int, default=3)
    parser.add_argument('--num-iters', type=int, default=30)
    parser.add_argument('--model-type', type=str, default='variational_sns')
    parser.add_argument('--save-histories', type=bool, default=False)
    parser.add_argument('--xla-allocator-platform', type=bool, default=False)
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--demixer-checkpoint', type=str,
        default='~/mbcs_grids/denoisers/seq_unet_50k_ai203_v2.ckpt')
    args = parser.parse_args(argseq)

    if args.xla_allocator_platform:
        print('Setting XLA_PYTHON_CLIENT_ALLOCATOR to PLATFORM')
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    return args




if __name__ == "__main__":

    args = parse_fit_options(sys.argv[1:])

    # load denoiser and denoise PSCs
    denoiser = NeuralDemixer(path=args.demixer_checkpoint, device='cpu')
    data_dict = np.load(args.dataset_path, allow_pickle=True)

    # flip psc so that deflections are positive
    psc = data_dict['psc']
    if np.sum(psc) < 0:
        psc = -psc

    den_psc = denoise_pscs_in_batches(psc, denoiser)
    del denoiser
    
    
#    stim, grid, loc_map, idx_map = make_grid_dataset(
#        data_dict['I'], data_dict['L'])
#    model = denoise_grid(args.model_type, fit_options,
#                         den_psc, stim)
#

    print('Finished denoising.')
    stims, pscs, den_pscs, Is, Ls, loc_maps, idx_maps = separate_data_by_plane(psc, den_psc, data_dict['I'], data_dict['L'])

    print('Running models on each plane.')
    plane_models = denoise_grid_planewise(args.minimax_spk_prob,
        args.num_iters, den_pscs, stims)

    results = dict(pscs=pscs,
                   den_pscs=den_pscs,
                   stim_mats=stims,
                   Is=Is,
                   Ls=Ls,
                   loc_maps=loc_maps,
                   idx_maps=idx_maps,
                   models=plane_models,
                   )

    print('Saving results.')
    dataset_name = os.path.basename(args.dataset_path).split('.')[0]
    outpath = args.outpath
    if args.outpath is None:
        outpath = dataset_name + '_singlespot_' + 'msp=%f' % args.minimax_spike_prob + '_msc=%d' % args.minimum_spike_count + '_results.npz'
    
    # save to numpy archive
    np.savez_compressed(outpath, **results)
    print('Saved singlespot results dictionary to %s .' % outpath)
