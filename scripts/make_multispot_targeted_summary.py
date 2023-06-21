import argparse
from circuitmap import NeuralDemixer
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import os

import subtractr.utils as utils

import argparse

def plot_before_after_spatial(mu_with, mu_without, targets, fig=None):
    if fig is None:
        fig = plt.figure(figsize=(5,5), dpi=300, facecolor='white')
    else:
        plt.sca(fig.gca())

    connected_idxs_without = np.where(mu_without > 0)[0]
    connected_idxs_with = np.where(mu_with > 0)[0]

    # set background to gray
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_facecolor('gray')

    # plot targeted locations as white circles
    plt.scatter(targets[:, 0], targets[:, 1], facecolors='none', edgecolors='white', s=50)

    # fill in neurons we find as connected without subtraction
    plt.scatter(targets[connected_idxs_without, 0], targets[connected_idxs_without, 1],
        facecolors='blue', edgecolors='white', s=50)
    # fill in neurons we find as connected with subtraction
    plt.scatter(targets[connected_idxs_with, 0], targets[connected_idxs_with, 1],
        facecolors='red', edgecolors='white', s=50)

    return fig

def compute_waveforms_by_power(powers, raw_traces, model_state):
    num_powers = len(np.unique(powers))
    timesteps = raw_traces.shape[1]
    mu = model_state['mu']
    num_neurons = mu.shape[0]

    waveforms = np.zeros((num_neurons, num_powers, timesteps))
    for pidx, power in enumerate(np.unique(powers)):
        these_trials = powers == power
        lam_curr = model_state['lam'][:, these_trials]
        raw_psc_curr = raw_traces[these_trials, :]
        waveforms[:, pidx, :] = utils.estimate_spike_waveforms(lam_curr, raw_psc_curr)
    return waveforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='.')
    parser.add_argument('--results_path', type=str,)
    parser.add_argument('--stim_start_idx', type=int, default=100)
    parser.add_argument('--stim_end_idx', type=int, default=200)

    args = parser.parse_args()

    # convert indices to ms
    stim_start_ms = args.stim_start_idx * 0.05
    stim_end_ms = args.stim_end_idx * 0.05

    R = utils.ComparisonResults(results_path=args.results_path)
    dset_name = os.path.basename(args.results_path)
    # plot before and after subtraction CAVIaR inferred connections
    fig1 = plot_before_after_spatial(
        R.get_multispot_weights(subtracted=True),
        R.get_multispot_weights(subtracted=False),
        R.get_targets()
    )

    # plot summary of subtraction by power
    pscs = R.get_multispot_pscs(subtracted=False)
    ests = R.get_multispot_ests()
    subtracted = R.get_multispot_pscs(subtracted=True)
    demixed = R.get_multispot_demixed(subtracted=True)
    powers = np.max(R.get_stim_mat(multispot=True, singlespot=False), axis=0)
    fig2 = utils.plot_subtraction_by_power(pscs, ests, subtracted, demixed, powers,
        stim_start_ms=stim_start_ms, stim_end_ms=stim_end_ms)

    # Estimate spike waveforms for each power
    fig3, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=300, facecolor='white')
    ss_powers = np.max(R.get_stim_mat(multispot=True, singlespot=False), axis=0)

    ss_waveforms_with_nps = compute_waveforms_by_power(ss_powers,
        R.get_multispot_pscs(subtracted=True), 
        R.get_multispot_model_state(subtracted=True))

    ss_waveforms_without_nps = compute_waveforms_by_power(ss_powers,
        R.get_multispot_pscs(subtracted=False), 
        R.get_multispot_model_state(subtracted=False))
    tsteps = np.arange(ss_waveforms_with_nps.shape[2]) * 0.05

    axs[0].plot(tsteps, ss_waveforms_without_nps[:, -1, :].T)
    axs[0].set_title('waveforms, no subtraction')

    axs[1].plot(tsteps, ss_waveforms_with_nps[:, -1, :].T)
    axs[1].set_title('waveforms, with subtraction')

    # draw vertical lines at stimulus onset and offset
    for ax in axs:
        ax.axvline(stim_start_ms, color='grey', linestyle='--')
        ax.axvline(stim_end_ms, color='grey', linestyle='--')

    outpath = os.path.join(args.save_path,
        dset_name + '_summary.pdf')
    print('saving figure to: %s' % outpath)
    pdf = matplotlib.backends.backend_pdf.PdfPages(outpath)
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.savefig(fig3)
    pdf.close()

    