import argparse
import numpy as np
import subtractr
import circuitmap as cm
import _pickle as cpickle  # pickle compression
import bz2
import jax.random as jrand

from datetime import date
from tqdm import tqdm

from subtractr.photocurrent_sim import add_photocurrents_to_expt
from subtractr.utils import add_subtraction_args
from circuitmap.simulation import simulate_continuous_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connection_prob')
    parser.add_argument('--spont_rate')
    parser.add_argument('--ntars')
    parser.add_argument('--token')
    parser.add_argument('--save_path')
    parser.add_argument('--n_sims_per_freq', default=10, type=int)

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

    # add subtraction parameters
    parser = utils.add_subtraction_args(parser)
    args = parser.parse_args()

    ntars = int(args.ntars)
    spont_rate = float(args.spont_rate)
    connection_prob = float(args.connection_prob)
    stim_freq = float(args.stim_freq)
    token = args.token
    n_sims = int(args.n_sims)

    N = 300
    nreps = 1
    trials = 2000
    sampling_freq = 20000
    demix = NeuralDemixer(path=args.demixer, device='cpu')
    expt_len = int(np.ceil(trials/stim_freq) * sampling_freq)
    ground_truth_eval_batch_size = 100

    results = {}
    
    # intialize random key
    key = jrand.PRNGKey(0)

    # loop over stim_freq inclusive of both max and min
    for stim_freq in np.arange(args.stim_freq_min, args.stim_freq_max + args.stim_freq_step, args.stim_freq_step):
        results['stim_freq'] = {}
        for i in tqdm(range(n_sims), leave=True):
            expt = simulate_continuous_experiment(N=N, H=ntars, nreps=nreps, spont_rate=spont_rate,
                                                connection_prob=connection_prob, stim_freq=stim_freq, expt_len=expt_len,
                                                ground_truth_eval_batch_size=ground_truth_eval_batch_size)

            # add photocurrents to the simulated experiment
            expt = add_photocurrents_to_expt(key, expt, stim_dur_ms=args.stim_dur_ms,
                prior_context=args.prior_context, response_length=args.response_length,)

            # run subtraction
            est = subtractr.low_rank.estimate_photocurrents_by_batches(
                expt['obs_with_photocurrents'], stim_start=args.stim_start, stim_end=args.stim_end,
                constrain_V=args.constrain_V, batch_size=args.batch_size, extended_baseline=args.extended_baseline,)
            subtracted = expt['obs_with_photocurrents'] - est

            results['stim_freq']['trial_%i' % i] = {
                'obs_with_photocurrents': expt['obs_with_photocurrents'],
                'subtracted': subtracted,
                'opsin_expression': expt['opsin_expression'],
            }

    outpath = os.path.join(args.save_path, 'stim_freq_sweep_N%i_K%i_ntars%i_nreps%i_connprob%.3f_spontrate%i_stimfreq%i_' % (
        N, trials, ntars, nreps, connection_prob, spont_rate, stim_freq) + token + '_%s.pkl' % (date.today().__str__()))

    with bz2.BZ2File(outpath, 'wb') as savefile:
        cpickle.dump(results, savefile)
