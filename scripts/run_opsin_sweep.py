import argparse
import numpy as np
import subtractr
import circuitmap as cm
import pandas as pd
import _pickle as cpickle  # pickle compression
import bz2
import os
import jax
import jax.random as jrand

from datetime import date
from tqdm import tqdm

import subtractr.experiment_sim as expsim
from subtractr.utils import add_subtraction_args
from circuitmap.simulation import simulate_continuous_experiment

# enable jax 64 bit mode
jax.config.update("jax_enable_x64", True)

def run_detection_pipeline(traces, stim_mat, demixer, args):
    demixed = demixer(traces)
    N = stim_mat.shape[0]
    model = cm.Model(N)
    model.fit(demixed,
        stim_mat,
        method='caviar',
        fit_options={
            'msrmp': args.msrmp,
            'iters': args.iters,
            'save_histories': args.save_histories,
            'minimum_spike_count': args.minimum_spike_count}
    )
    return model.state['mu']
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connection_prob', default=0.1)
    parser.add_argument('--spont_rate', default=0.1)
    parser.add_argument('--ntars', default=10)
    parser.add_argument('--token', default='')
    parser.add_argument('--save_path', default='./')
    parser.add_argument('--n_sims_per_freq', default=10, type=int)
    parser.add_argument('--num_neurons', default=300, type=int)
    parser.add_argument('--num_trials', default=2000, type=int)

    # add photocurrent shape parameters
    parser.add_argument('--O_inf_min', type=float, default=0.3)
    parser.add_argument('--O_inf_max', type=float, default=1.0)
    parser.add_argument('--R_inf_min', type=float, default=0.3)
    parser.add_argument('--R_inf_max', type=float, default=1.0)
    parser.add_argument('--tau_o_min', type=float, default=3)
    parser.add_argument('--tau_o_max', type=float, default=30)
    parser.add_argument('--tau_r_min', type=float, default=3)
    parser.add_argument('--tau_r_max', type=float, default=30)

    # add photocurrent amount parameters
    parser.add_argument('--frac_pc_cells', type=float, default=0.01)
    parser.add_argument('--opsin_mean', type=float, default=0.3)
    parser.add_argument('--opsin_std', type=float, default=0.2)
    parser.add_argument('--stim_dur_ms', type=float, default=5.0)
    parser.add_argument('--pc_response_var', type=float, default=0.01)
    parser.add_argument('--sampling_freq', type=float, default=20000)
    parser.add_argument('--prior_context', type=int, default=100)
    parser.add_argument('--response_length', type=int, default=2000)

    # parameters to sweep stim_frequency
    parser.add_argument('--stim_freq', type=float, default=30)

    # add frac_pc_cells sweep parameters
    parser.add_argument('--frac_pc_cells_min', type=float, default=0.01)
    parser.add_argument('--frac_pc_cells_max', type=float, default=0.1)
    parser.add_argument('--frac_pc_cells_step', type=float, default=0.01)
    parser.add_argument('--num_sims_per_sweep', type=int, default=1)

    # demixer parameters
    parser.add_argument('--demixer_path', type=str)

    # caviar args
    parser = utils.add_caviar_args(parser=parser)

    # add subtraction parameters
    parser = add_subtraction_args(parser)
    args = parser.parse_args()

    # load demixer
    demixer = cm.NeuralDemixer(path=args.demixer_path, device='cpu')

    ntars = int(args.ntars)
    spont_rate = float(args.spont_rate)
    connection_prob = float(args.connection_prob)
    token = args.token

    nreps = 1
    sampling_freq = 20000
    ground_truth_eval_batch_size = 100

    results = pd.DataFrame(columns=['stim_freq', 'trial',
                                    'subtracted', 'original',
                                    'weights_subtracted',
                                    'weights_raw',
                                    'weights_true'])
    results['stim_freq'] = results['stim_freq'].astype('float')
    results['trial'] = results['trial'].astype('int')
    results['subtracted'] = results['subtracted'].astype('object')
    results['original'] = results['original'].astype('object')
    results['weights_subtracted'] = results['weights_subtracted'].astype('object')
    results['weights_raw'] = results['weights_raw'].astype('object')


    df_idx = 0

    # intialize random key
    key = jrand.PRNGKey(0)

    # sweep over frac_pc_cells
    for frac_pc_cells in np.arange(args.frac_pc_cells_min, args.frac_pc_cells_max + args.frac_pc_cells_step, args.frac_pc_cells_step):
        for i in tqdm(range(args.num_expts_per_freq), leave=True):
            expt_len = int(np.ceil(args.num_trials/stim_freq)
                           * args.sampling_freq)
            expt = simulate_continuous_experiment(N=args.num_neurons, H=ntars, nreps=nreps, spont_rate=spont_rate,
                                                  connection_prob=connection_prob, stim_freq=args.stim_freq, expt_len=expt_len,
                                                  ground_truth_eval_batch_size=ground_truth_eval_batch_size,
                                                  response_length=args.response_length,)

            # add photocurrents to the simulated experiment
            key = jrand.fold_in(key, i)
            expt = expsim.add_photocurrents_to_expt(key, expt,
                                             frac_pc_cells=frac_pc_cells,
                                             opsin_mean=args.opsin_mean,
                                             opsin_std=args.opsin_std,
                                             stim_dur_ms=args.stim_dur_ms,
                                             pc_response_var=args.pc_response_var,
                                             pc_window_len_ms=args.response_length,
                                             sampling_freq=args.sampling_freq,
                                             stim_freq=args.stim_freq,
                                             prior_context=args.prior_context,
                                             response_length=args.response_length,
                                             )

            # run subtraction
            est = subtractr.low_rank.estimate_photocurrents_by_batches(
                expt['obs_with_photocurrents'],
                stim_start=args.stim_start_idx,
                stim_end=args.stim_end_idx,
                constrain_V=args.constrain_V, batch_size=args.batch_size,
                rank=args.rank, subtract_baselines=False)

            # Subtract using the overlapping method
            subtracted_flat = expsim.subtract_overlapping_trials(expt['obs_with_photocurrents'], est,
                                                          prior_context=args.prior_context, stim_freq=args.stim_freq, sampling_freq=args.sampling_freq,
                                                          return_flat=True,)

            # re-fold subtracted trials into matrix
            subtracted = expsim.fold_overlapping(subtracted_flat, args.prior_context, args.response_length, args.sampling_freq, args.stim_freq)

            # Run demixing and CAVIaR without subtraction
            mu_without = run_detection_pipeline(expt['obs_with_photocurrents'],
                expt['stim_matrix'], demixer, args)

            # Run demixing and CAVIaR with subtraction
            mu_with = run_detection_pipeline(subtracted,
                expt['stim_matrix'], demixer, args)

            # add current results to dataframe
            results.loc[df_idx, 'stim_freq'] = stim_freq
            results.loc[df_idx, 'trial'] = i
            results.loc[df_idx, 'weights_subtracted'] = [mu_with]
            results.loc[df_idx, 'weights_raw'] = [mu_without]
            results.loc[df_idx, 'weights_true'] = [expt['weigthts']]

            # only save traces for the first trial
            if i == 0:
                results.loc[df_idx, 'subtracted'] = [subtracted]
                results.loc[df_idx, 'original'] = [expt['obs_with_photocurrents']]

            df_idx += 1

    outpath = os.path.join(args.save_path, 'stim_freq_sweep_N%i_K%i_ntars%i_nreps%i_connprob%.3f_spontrate%i_stimfreq%i_' % (
        args.num_neurons, args.num_trials, ntars, nreps, connection_prob, spont_rate, stim_freq) + token + '_%s.pkl' % (date.today().__str__()))

    with bz2.BZ2File(outpath, 'wb') as savefile:
        cpickle.dump(results, savefile)
