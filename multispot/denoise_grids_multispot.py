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
    den_psc_batched = [denoiser(batch, verbose=False) for batch in np.array_split(psc, num_batches, axis=0)]
    return np.concatenate(den_psc_batched)



def denoise_grid(minimax_spk_prob, iters, den_psc, stim):

    # create priors and model
    N,K = stim.shape
    model = cm.Model(N)

    # fit model and save
    model.fit(den_psc, stim, method='caviar',
        fit_options={'minimax_spk_prob':minimax_spk_prob, 'iters':iters})


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

    stim = data_dict['stim_matrix']
    model = denoise_grid(args.minimax_spk_prob,
        args.num_iters, den_psc, stim)
    results = dict(multispot_psc=psc,
        multispot_den_psc=den_psc,
        multispot_stim=stim,
        multispot_model=model)


    dataset_name = os.path.basename(args.dataset_path).split('.')[0]
    outpath = dataset_name + '_singlespot_' + 'msp=%.1f_' % args.minimax_spike_prob + args.model_type + '_results.npz'
    
    # save to numpy archive
    np.savez_compressed(outpath, **results)

    print('Saved multispot results dictionary to %s' % outpath)
