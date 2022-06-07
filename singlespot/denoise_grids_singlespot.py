#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import sys
import adaprobe
from adaprobe.psc_denoiser import NeuralDenoiser
from adaprobe.simulation import simulate
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


def make_priors_mbcs(N, K):
    beta_prior = 3e0 * np.ones(N)
    mu_prior = np.zeros(N)
    rate_prior = 1e-1 * np.ones(K)
    shape_prior = np.ones(K)

    priors = {
        'beta': beta_prior,
        'mu': mu_prior,
        'shape': shape_prior,
        'rate': rate_prior,
    }
    return priors


def make_priors_vsns(N, K):
    phi_prior = np.c_[0.125 * np.ones(N), 5 * np.ones(N)]
    phi_cov_prior = np.array(
        [np.array([[1e-1, 0], [0, 1e0]]) for _ in range(N)])
    alpha_prior = 0.15 * np.ones(N)
    beta_prior = 3e0 * np.ones(N)
    mu_prior = np.zeros(N)
    sigma_prior = np.ones(N)
    sigma = 1

    priors_vsns = {
        'alpha': alpha_prior,
        'beta': beta_prior,
        'mu': mu_prior,
        'phi': phi_prior,
        'phi_cov': phi_cov_prior,
        'shape': 1.,
        'rate': sigma**2,
    }

    return priors_vsns


def denoise_grid(model_type, fit_options, den_psc, stim):
    all_models = []

    if model_type == 'mbcs':
        prior_fn = make_priors_mbcs
        method = 'mbcs_spike_weighted_var_with_outliers'
    elif model_type == 'variational_sns':
        prior_fn = make_priors_vsns
        method = 'cavi_sns'
    else:
        raise ValueError("invalid model type...")

    N, K = stim.shape
    priors = prior_fn(N, K)
    model_params = {'N': N, 'model_type': model_type, 'priors': priors}
    model = adaprobe.Model(**model_params)

    # fit model and save
    model.fit(psc, stim, fit_options=fit_options, method=method)

    return model


def parse_fit_options(argseq):
    parser = argparse.ArgumentParser(
        description='MBCS for Grid Denoising')
    parser.add_argument('--minimax-spike-prob', type=float, default=0.1)
    parser.add_argument('--minimum-spike-count', type=int, default=3)
    parser.add_argument('--num-iters', type=int, default=30)
    parser.add_argument('--model-type', type=str, default='variational_sns')
    parser.add_argument('--save-histories', type=bool, default=False)
    parser.add_argument('--xla-allocator-platform', type=bool, default=False)
    parser.add_argument('--dataset-path', type=str)
    args = parser.parse_args(argseq)

    if args.xla_allocator_platform:
        print('Setting XLA_PYTHON_CLIENT_ALLOCATOR to PLATFORM')
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    # parameters you can vary
    iters = args.num_iters
    minimax_spike_prob = args.minimax_spike_prob
    minimum_spike_count = args.minimum_spike_count

    # fit options for mbcs
    seed = 1
    y_xcorr_thresh = 1e-2
    max_penalty_iters = 50
    warm_start_lasso = True
    verbose = False
    num_mc_samples_noise_model = 100
    noise_scale = 0.5
    init_spike_prior = 0.5
    num_mc_samples = 500
    penalty = 2
    max_lasso_iters = 1000
    scale_factor = 0.75
    constrain_weights = 'positive'
    orthogonal_outliers = True
    lam_mask_fraction = 0.0
    delay_spont_estimation = 2

    fit_options_mbcs = {
        'iters': iters,
        'num_mc_samples': num_mc_samples,
        'penalty': penalty,
        'max_penalty_iters': max_penalty_iters,
        'max_lasso_iters': max_lasso_iters,
        'scale_factor': scale_factor,
        'constrain_weights': constrain_weights,
        'y_xcorr_thresh': y_xcorr_thresh,
        'seed': seed,
        'verbose': verbose,
        'warm_start_lasso': warm_start_lasso,
        'minimum_spike_count': minimum_spike_count,
        'minimum_maximal_spike_prob': minimax_spike_prob,
        'noise_scale': noise_scale,
        'init_spike_prior': init_spike_prior,
        'orthogonal_outliers': orthogonal_outliers,
        'lam_mask_fraction': lam_mask_fraction,
        'delay_spont_estimation': delay_spont_estimation
    }

    # fit options for vsns
    iters = args.num_iters
    minimum_spike_count = args.minimum_spike_count
    minimax_spike_prob = args.minimax_spike_prob
    sigma = 1.
    seed = 1
    y_xcorr_thresh = 1e-2
    max_penalty_iters = 50
    warm_start_lasso = True
    verbose = False
    num_mc_samples_noise_model = 100
    noise_scale = 0.5
    init_spike_prior = 0.5
    num_mc_samples = 500
    penalty = 1e1
    max_lasso_iters = 1000
    scale_factor = 0.75
    constrain_weights = 'positive'
    orthogonal_outliers = True
    lam_mask_fraction = 0.025

    fit_options_vsns = {
        'iters': iters,
        'num_mc_samples': num_mc_samples,
        'y_xcorr_thresh': y_xcorr_thresh,
        'seed': seed,
        'phi_thresh': None,
        'phi_thresh_delay': -1,
        'learn_noise': True,
        'minimax_spk_prob': 0.2,
        'scale_factor': 0.75,
        'penalty': penalty,
        'save_histories': args.save_histories
    }

    if args.model_type == 'mbcs':
        return fit_options_mbcs, args
    elif args.model_type == 'variational_sns':
        return fit_options_vsns, args
    else:
        raise ValueError("Unknown argument for model type")


if __name__ == "__main__":

    fit_options, args = parse_fit_options(sys.argv[1:])

    # load denoiser and denoise PSCs
    denoiser = NeuralDenoiser(path='~/mbcs_grids/denoisers/seq_unet_50k_ai203_v2.ckpt')
    data_dict = np.load(args.dataset_path, allow_pickle=True)

    # flip psc so that deflections are positive
    psc = data_dict['psc']
    if np.sum(psc) < 0:
        psc = -psc

    den_psc = denoise_pscs_in_batches(psc, denoiser)

    # free denoiser from memory
    denoiser.denoiser.to('cpu')
    del denoiser
    gc.collect()
    torch.cuda.empty_cache()
    
    
    stim, grid, loc_map, idx_map = make_grid_dataset(
        data_dict['I'], data_dict['L'])
    model = denoise_grid(args.model_type, fit_options,
                         den_psc, stim)

    results = dict(psc=psc,
                   den_psc=den_psc,
                   stim=stim,
                   I=data_dict['I'],
                   L=data_dict['L'],
                   idx_map=idx_map,
                   loc_map=loc_map,
                   model=model
                   )
    dataset_name = os.path.basename(args.dataset_path).split('.')[0]
    outpath = dataset_name + '_singlespot_' + 'msp=%.1f_' % args.minimax_spike_prob + args.model_type + '_results.npz'
    
    # save to numpy archive
    np.savez_compressed(outpath, **results)
    print('Saved singlespot results dictionary to %s' % outpath)
