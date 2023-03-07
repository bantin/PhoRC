import argparse
import numpy as np
import subtractr
import circuitmap as cm
import pandas as pd
import _pickle as cpickle  # pickle compression
import bz2
import os
import jax.random as jrand

from datetime import date
from tqdm import tqdm

from subtractr.photocurrent_sim import add_photocurrents_to_expt
from subtractr.utils import add_subtraction_args
from circuitmap.simulation import simulate_continuous_experiment

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
    parser.add_argument('--frac_pc_cells', type=float, default=0.1)
    parser.add_argument('--opsin_mean', type=float, default=0.5)
    parser.add_argument('--opsin_std', type=float, default=0.2)
    parser.add_argument('--stim_dur_ms', type=float, default=5.0)
    parser.add_argument('--prior_context', type=int, default=100)
    parser.add_argument('--pc_response_var', type=float, default=0.1)
    parser.add_argument('--response_length', type=float, default=900)


    # parameters to sweep stim_frequency
    parser.add_argument('--stim_freq_min', type=float, default=10)
    parser.add_argument('--stim_freq_max', type=float, default=50)
    parser.add_argument('--stim_freq_step', type=float, default=10)
    parser.add_argument('--num_expts_per_freq', type=int, default=1)

    # add subtraction parameters
    parser = add_subtraction_args(parser)
    args = parser.parse_args()

    ntars = int(args.ntars)
    spont_rate = float(args.spont_rate)
    connection_prob = float(args.connection_prob)
    token = args.token

    nreps = 1
    sampling_freq = 20000
    ground_truth_eval_batch_size = 100

    results = pd.DataFrame(columns=['stim_freq', 'trial', 'obs_with_photocurrents', 'subtracted', 'original', 'opsin_expression'])
    results['obs_with_photocurrents'] = results['obs_with_photocurrents'].astype(object)
    results['subtracted'] = results['subtracted'].astype(object)
    results['original'] = results['original'].astype(object)
    results['opsin_expression'] = results['opsin_expression'].astype(object)
    df_idx = 0
    
    # intialize random key
    key = jrand.PRNGKey(0)

    # loop over stim_freq inclusive of both max and min

    for stim_freq in np.arange(args.stim_freq_min, args.stim_freq_max + args.stim_freq_step, args.stim_freq_step):
        for i in tqdm(range(args.num_expts_per_freq), leave=True):
            expt_len = int(np.ceil(args.num_trials/stim_freq) * sampling_freq)
            expt = simulate_continuous_experiment(N=args.num_neurons, H=ntars, nreps=nreps, spont_rate=spont_rate,
                                                connection_prob=connection_prob, stim_freq=stim_freq, expt_len=expt_len,
                                                ground_truth_eval_batch_size=ground_truth_eval_batch_size)

            # add photocurrents to the simulated experiment
            key = jrand.fold_in(key, i)
            expt = add_photocurrents_to_expt(key, expt, stim_dur_ms=args.stim_dur_ms,
                prior_context=args.prior_context, response_length=args.response_length,)

            # run subtraction
            est = subtractr.low_rank.estimate_photocurrents_by_batches(
                expt['obs_with_photocurrents'], 
                stim_start=args.stim_start_idx, 
                stim_end=args.stim_end_idx,
                constrain_V=args.constrain_V, batch_size=args.batch_size,
                extended_baseline=args.extended_baseline,
                rank=args.rank,)
            subtracted = expt['obs_with_photocurrents'] - est
            orig_pscs = expt['obs_responses']

            # add current results to dataframe
            results.loc[df_idx, 'stim_freq'] = stim_freq
            results.loc[df_idx, 'trial'] = i
            results.loc[df_idx, 'obs_with_photocurrents'] = [expt['obs_with_photocurrents']]
            results.loc[df_idx, 'subtracted'] = [subtracted]
            results.loc[df_idx, 'original'] = [orig_pscs]
            results.loc[df_idx, 'opsin_expression'] = [expt['opsin_expression'][:,None]]
            df_idx += 1

    outpath = os.path.join(args.save_path, 'stim_freq_sweep_N%i_K%i_ntars%i_nreps%i_connprob%.3f_spontrate%i_stimfreq%i_' % (
        args.num_neurons, args.num_trials, ntars, nreps, connection_prob, spont_rate, stim_freq) + token + '_%s.pkl' % (date.today().__str__()))

    with bz2.BZ2File(outpath, 'wb') as savefile:
        cpickle.dump(results, savefile)
