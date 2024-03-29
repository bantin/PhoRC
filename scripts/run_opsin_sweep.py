import argparse
import numpy as np
import jax.numpy as jnp
import phorc
import circuitmap as cm
import pandas as pd
import _pickle as cpickle
import bz2
import os
import jax
import jax.random as jrand
import itertools

from datetime import date
from tqdm import tqdm

from phorc.simulation import experiment_sim as expsim
from phorc.utils import add_subtraction_args, add_caviar_args
from circuitmap.simulation import simulate_continuous_experiment

# enable jax 64 bit mode
jax.config.update("jax_enable_x64", True)

# seed numpy
np.random.seed(0)

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
    parser.add_argument('--num_neurons', default=100, type=int)
    parser.add_argument('--num_trials', default=2000, type=int)

    parser.add_argument('--tau_delta_min', type=float, default=60) # match e->e connections
    parser.add_argument('--tau_delta_max', type=float, default=120)
    parser.add_argument('--weight_lower', type=float, default=5)
    parser.add_argument('--weight_upper', type=float, default=10)
    parser.add_argument('--strong_weight_lower', type=float, default=20)
    parser.add_argument('--strong_weight_upper', type=float, default=30)
    parser.add_argument('--gamma_beta', type=float, default=25) # distribution of latencies
    parser.add_argument('--noise_std', type=float, default=0.01) # noise in PSCs

    # add params to sweep over min latency
    parser.add_argument('--min_latency_min', type=int, default=60) # min latency of PSCs at max power
    parser.add_argument('--min_latency_max', type=int, default=100)
    parser.add_argument('--min_latency_step', type=int, default=10)


    # add photocurrent amount parameters
    parser.add_argument('--opsin_mean', type=float, default=0.3)
    parser.add_argument('--opsin_std', type=float, default=0.2)
    parser.add_argument('--stim_dur_ms', type=float, default=5.0)
    parser.add_argument('--pc_response_var', type=float, default=0.01)
    parser.add_argument('--sampling_freq', type=float, default=20000)
    parser.add_argument('--prior_context', type=int, default=100)
    parser.add_argument('--response_length', type=int, default=900)

    # add frac_pc_cells sweep parameters
    parser.add_argument('--frac_pc_cells_min', type=float, default=0.01)
    parser.add_argument('--frac_pc_cells_max', type=float, default=0.1)
    parser.add_argument('--frac_pc_cells_step', type=float, default=0.01)
    parser.add_argument('--num_sims_per_sweep', type=int, default=1)

    # add stim_freq sweep parameters
    parser.add_argument('--stim_freq_min', type=float, default=10)
    parser.add_argument('--stim_freq_max', type=float, default=50)
    parser.add_argument('--stim_freq_step', type=float, default=10)

    # add stim_end_idx sweep parameters
    parser.add_argument('--stim_end_idx_min', type=int, default=160)
    parser.add_argument('--stim_end_idx_max', type=int, default=200)
    parser.add_argument('--stim_end_idx_step', type=int, default=10)

    # add option to automatically set stim_end_idx to match min_latency
    parser.add_argument('--autoset_stim_end_idx', action='store_true')
    parser.set_defaults(autoset_stim_end_idx=True)

    # demixer parameters
    parser.add_argument('--demixer_path', type=str)
    parser.add_argument('--demixer_response_length', type=int, default=900)

    # add option to use network for photocurrent subtraction
    parser.add_argument('--use_network', action='store_true')
    parser.add_argument('--network_path', type=str, default=None)
    parser.set_defaults(use_network=False)

    parser.add_argument('--save_expts', action='store_true')
    parser.set_defaults(save_expts=False)


    # caviar args
    parser = add_caviar_args(parser=parser)

    # add subtraction parameters
    parser = add_subtraction_args(parser)
    args = parser.parse_args()

    # load demixer
    demixer = cm.NeuralDemixer(path=args.demixer_path, device='cpu')

    # load network
    if args.use_network:
        network = phorc.phorc.load_from_checkpoint(args.network_path)

    ntars = int(args.ntars)
    spont_rate = float(args.spont_rate)
    connection_prob = float(args.connection_prob)
    token = args.token

    nreps = 1
    sampling_freq = 20000
    ground_truth_eval_batch_size = 100

    results = pd.DataFrame(columns=['stim_freq', 'trial',
                                    'frac_pc_cells',
                                    'opsin_expression',
                                    'subtracted',
                                    'original',
                                    'ground_truth',
                                    'stim_matrix',
                                    'weights_subtracted',
                                    'weights_raw',
                                    'weights_true',
                                    'weights_oracle',
                                    'min_latency',
                                    'use_network',
                                    'mse',
                                    'true_photocurrents',])

    df_idx = 0

    # intialize random key
    key = jrand.PRNGKey(0)

    # initialize list for saving experiments
    expts = []

    # sweep over frac_pc_cells and min_latency using itertools.product
    latencies = np.arange(args.min_latency_min, args.min_latency_max + args.min_latency_step, args.min_latency_step)
    frac_pc_cells_vals = np.arange(args.frac_pc_cells_min, args.frac_pc_cells_max + args.frac_pc_cells_step, args.frac_pc_cells_step)
    stim_freqs = np.arange(args.stim_freq_min, args.stim_freq_max + args.stim_freq_step, args.stim_freq_step)

    for frac_pc_cells, min_latency, stim_freq in itertools.product(frac_pc_cells_vals, latencies, stim_freqs):
        for i in tqdm(range(args.num_sims_per_sweep), leave=True):
            expt_len = int(np.ceil(args.num_trials/stim_freq)
                           * args.sampling_freq)
            expt = simulate_continuous_experiment(N=args.num_neurons, H=ntars, nreps=nreps, spont_rate=spont_rate,
                                                  connection_prob=connection_prob, stim_freq=stim_freq, expt_len=expt_len,
                                                  ground_truth_eval_batch_size=ground_truth_eval_batch_size,
                                                  response_length=args.response_length,
                                                  tau_delta_min=args.tau_delta_min,
                                                  tau_delta_max=args.tau_delta_max,
                                                  weight_lower=args.weight_lower,
                                                  weight_upper=args.weight_upper,
                                                  strong_weight_lower=args.strong_weight_lower,
                                                  strong_weight_upper=args.strong_weight_upper,
                                                  gamma_beta=args.gamma_beta,
                                                  min_latency=min_latency,
                                                  noise_std=args.noise_std,)
            # convert all jax arrays to numpy arrays inside of expt dict
            for k, v in expt.items():
                expt[k] = np.array(v)

            if args.save_expts:
                expts.append(expt)

            # add photocurrents to the simulated experiment
            key = jrand.fold_in(key, i)
            expt = expsim.add_photocurrents_to_expt(key, expt,
                                             frac_pc_cells=frac_pc_cells,
                                             opsin_mean=args.opsin_mean,
                                             opsin_std=args.opsin_std,
                                             stim_dur_ms=args.stim_dur_ms,
                                             pc_response_var=args.pc_response_var,
                                             sampling_freq=args.sampling_freq,
                                             stim_freq=stim_freq,
                                             prior_context=args.prior_context,
                                             response_length=args.response_length,
                                             )

            # run subtraction
            if args.use_network:
                est = network(expt['obs_with_photocurrents'][:, 0:args.demixer_response_length])

            else:
                est = phorc.estimate(
                    expt['obs_with_photocurrents'][:, 0:args.demixer_response_length],
                    window_start_idx=args.window_start_idx,
                    window_end_idx=args.window_end_idx,
                    batch_size=args.batch_size,
                    rank=args.rank, subtract_baseline=True)


            # To simplify simulations, we wont' use the overlapping subtraction method
            # for now. 
            obs = expt['obs_with_photocurrents'][:, :args.demixer_response_length]
            subtracted = expt['obs_with_photocurrents'][:, 0:args.demixer_response_length] - est

            # Compute mse between subtracted and original
            subtracted_oracle = obs - expt['true_photocurrents'][:, 0:args.demixer_response_length]
            mse = np.mean((subtracted - subtracted_oracle)**2)

            # Run demixing and CAVIaR without subtraction
            mu_without = run_detection_pipeline(obs,
                expt['stim_matrix'], demixer, args)

            # Run demixing and CAVIaR with subtraction
            mu_with = run_detection_pipeline(subtracted,
                expt['stim_matrix'], demixer, args)

            # run demixing on caviar on observations before photocurrents are added
            # this gives oracle
            mu_oracle = run_detection_pipeline(expt['obs_responses'],
                expt['stim_matrix'], demixer, args)

            # add current results to dataframe
            results.loc[df_idx, 'stim_freq'] = stim_freq
            results.loc[df_idx, 'trial'] = i
            results.loc[df_idx, 'weights_subtracted'] = mu_with
            results.loc[df_idx, 'weights_raw'] = mu_without
            results.loc[df_idx, 'weights_oracle'] = mu_oracle
            results.loc[df_idx, 'weights_true'] = expt['weights']
            results.loc[df_idx, 'frac_pc_cells'] = frac_pc_cells
            results.loc[df_idx, 'opsin_expression'] = expt['opsin_expression']
            results.loc[df_idx, 'min_latency'] = min_latency
            results.loc[df_idx, 'use_network'] = args.use_network
            results.loc[df_idx, 'mse'] = mse

            # only save traces for the first trial
            if i == 0:
                results.loc[df_idx, 'stim_matrix'] = expt['stim_matrix']
                results.loc[df_idx, 'subtracted'] = subtracted
                results.loc[df_idx, 'original'] = obs
                results.loc[df_idx, 'ground_truth'] = expt['obs_responses'][:, :args.demixer_response_length]
                results.loc[df_idx, 'true_photocurrents'] = expt['true_photocurrents']

            df_idx += 1

    if args.use_network:
        outname = 'subtraction_sweep_network_N%i_K%i_ntars%i_nreps%i_connprob%.3f_spontrate%i_stimfreqmax%i_numsims%i' % (
            args.num_neurons, args.num_trials, ntars, nreps, connection_prob, spont_rate, args.stim_freq_max, args.num_sims_per_sweep) + token + '_%s.pkl' % (date.today().__str__())
    else:
        outname = 'subtraction_sweep_N%i_K%i_ntars%i_nreps%i_connprob%.3f_spontrate%i_stimfreqmax%i_numsims%i' % (
            args.num_neurons, args.num_trials, ntars, nreps, connection_prob, spont_rate, args.stim_freq_max, args.num_sims_per_sweep) + token + '_%s.pkl' % (date.today().__str__())
    outpath = os.path.join(args.save_path, outname)
    with bz2.BZ2File(outpath, 'wb') as savefile:
        cpickle.dump(results, savefile)

    if args.save_expts:
        exp_outname = 'expts_N%i_K%i_ntars%i_nreps%i_connprob%.3f_spontrate%i_stimfreqmax%i_numsims%i' % (
            args.num_neurons, args.num_trials, ntars, nreps, connection_prob, spont_rate, args.stim_freq_max, args.num_sims_per_sweep) + token + '_%s.pkl' % (date.today().__str__())
        exp_outpath = os.path.join(args.save_path, exp_outname)
        with bz2.BZ2File(exp_outpath, 'wb') as savefile:
            cpickle.dump(expts, savefile)
