import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import subtractr

pc_fracs = np.arange(0.1, 1.0 + 0.1, 0.1)
iid_noise_levels = np.arange(0.01, 0.1 + 0.01, 0.01)
num_runs = 5

net = subtractr.Subtractr.load_from_checkpoint('../subtractr/lightning_logs/version_500470/checkpoints/epoch=999-step=3125000.ckpt')
net = net.eval()
if torch.cuda.is_available():
    net = net.to('cuda')

def create_formatted_df():
    df = pd.DataFrame(columns=['noise_level', 'pc_frac', 'method', 'run', 'mse', 'inputs', 'true_pscs'])
    df['noise_level'] = df['noise_level'].astype(float)
    df['pc_frac'] = df['pc_frac'].astype(float)
    df['method'] = df['method'].astype(str)
    df['run'] = df['run'].astype(int)
    df['mse'] = df['mse'].astype(float)
    df['true_pscs'] = df['true_pscs'].astype(object)
    df['inputs'] = df['inputs'].astype(object)

    return df


low_rank_mse = np.zeros((len(pc_fracs), len(iid_noise_levels), num_runs))
network_mse = np.zeros((len(pc_fracs), len(iid_noise_levels), num_runs))
num_runs = 100
df = create_formatted_df()
for i, frac in enumerate(pc_fracs):
    for j, noise in enumerate(iid_noise_levels):

        argstr = ("--num_train 10"
            # " --linear_onset_frac 0.5"
            " --pc_scale_min 0.1 --pc_scale_max 0.8"
            " --psc_scale_min 0.01 --psc_scale_max 0.5"
            " --min_pc_fraction %f --max_pc_fraction %f"
            " --max_pc_fraction 1.0"
            " --onset_latency_ms 0.0 --num_test %d"
            " --normalize none"
            " --onset_jitter_ms 1.0"
            " --gp_scale_max 0.01"
            " --gp_scale_min 0.01"
            " --num_traces_per_expt 64"
            " --iid_noise_std_min %f"
            " --iid_noise_std_max %f" % (frac, frac, num_runs, noise, noise))
        argstr = argstr.split(" ")
        args = subtractr.pc_subtractr_network.parse_args(argstr)

        dummy_net = subtractr.Subtractr(**vars(args))
        dummy_net.generate_training_data(args)

        for run_idx in range(num_runs):
            # each run here is its own simulated experiment
            inputs = np.array(dummy_net.test_expts[0][run_idx])
            true_pscs = inputs - np.array(dummy_net.test_expts[1][run_idx])

            # subtract photocurrent using trained net
            est = net(np.array(inputs))
            subtracted = inputs - est
            new_row = create_formatted_df()
            new_row.loc[0] = {'noise_level': noise, 'pc_frac': frac, 'method': 'network',
                'run': run_idx, 'mse': np.mean((est - true_pscs)**2),
                'inputs': inputs, 'true_pscs': true_pscs}
            df = pd.concat([df, new_row], ignore_index=True)


            # calculate mse using low-rank model
            est = subtractr.low_rank.estimate_photocurrents_baseline(inputs,
                None, separate_by_power=False, stepwise_constrain_V=False)
            subtracted = inputs - est
            new_row = create_formatted_df()
            new_row.loc[0] = {'noise_level': noise, 'pc_frac': frac, 'method': 'low-rank',
                'run': run_idx, 'mse': np.mean((est - true_pscs)**2),
                'inputs': inputs, 'true_pscs': true_pscs}
            df = pd.concat([df, new_row], ignore_index=True)

df.to_pickle('synthetic_validation.pkl', compression={'method': 'gzip', 'compresslevel': 1})