from ast import Mult
from multiprocessing.sharedctypes import Value
from typing import Union
import numpy as np
import circuitmap as cm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import time
import sys

import argparse
from argparse import ArgumentParser
import photocurrent_sim
import jax
import jax.random as jrand
from itertools import cycle
from backbones import MultiTraceConv, SetTransformer, MultiTraceConvAttention, SingleTraceConv
jax.config.update('jax_platform_name', 'cpu')


class Subtractr(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Subtractr")
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--batch_size', type=int, default=8)
        parser.add_argument('--num_train', type=int, default=1000)
        parser.add_argument('--num_test', type=int, default=100)
        parser.add_argument('--trial_dur', type=int, default=900)
        parser.add_argument('--num_traces_per_expt', type=int, default=50)
        parser.add_argument('--photocurrent_scale_min',
                            type=float, default=0.01)
        parser.add_argument('--photocurrent_scale_max',
                            type=float, default=1.1)
        parser.add_argument('--pc_scale_min', type=float, default=0.1)
        parser.add_argument('--pc_scale_max', type=float, default=10.0)
        parser.add_argument('--gp_scale_min', type=float, default=0.01)
        parser.add_argument('--gp_scale_max', type=float, default=0.2)
        parser.add_argument('--iid_noise_scale_min', type=float, default=0.01)
        parser.add_argument('--iid_noise_scale_max', type=float, default=0.1)
        parser.add_argument('--min_pc_fraction', type=float, default=0.5)
        parser.add_argument('--max_pc_fraction', type=float, default=1.0)
        parser.add_argument('--gp_lengthscale', type=float, default=50.0)

        # whether we use the LS solver at the end of forward pass
        parser.add_argument('--use_ls_solve', action='store_true')
        parser.add_argument('--no_use_ls_solve', dest='use_ls_solve', action='store_false')
        parser.set_defaults(use_ls_solve=False)
        
        # whether we add a gp to the target waveforms
        parser.add_argument('--add_target_gp', action='store_true')
        parser.add_argument('--no_add_target_gp', dest='add_target_gp', action='store_false')
        parser.set_defaults(add_target_gp=True)
        parser.add_argument('--target_gp_lengthscale', default=25)
        parser.add_argument('--target_gp_scale', default=0.01)

        # whether we use the linear onset in the training data
        parser.add_argument('--linear_onset_frac', type=float, default=0.5)

        # photocurrent shape args
        parser.add_argument('--O_inf_min', type=float, default=0.3)
        parser.add_argument('--O_inf_max', type=float, default=1.0)
        parser.add_argument('--R_inf_min', type=float, default=0.3)
        parser.add_argument('--R_inf_max', type=float, default=1.0)
        parser.add_argument('--tau_o_min', type=float, default=5)
        parser.add_argument('--tau_o_max', type=float, default=7)
        parser.add_argument('--tau_r_min', type=float, default=26)
        parser.add_argument('--tau_r_max', type=float, default=29)

        # photocurrent timing args
        parser.add_argument('--onset_jitter_ms', type=float, default=1.0)
        parser.add_argument('--onset_latency_ms', type=float, default=0.2)

        # architecture type
        parser.add_argument('--model_type', type=str, default='MultiTraceConv')

        # convolutional args. 
        parser.add_argument('--down_filter_sizes', nargs=4, type=int, default=(16, 32, 64, 128))
        parser.add_argument('--up_filter_sizes', nargs=4, type=int, default=(64, 32, 16, 4))

        # SetTransformer args
        parser.add_argument('--dim_input', type=int, default=900)
        parser.add_argument('--num_inds', type=int, default=64)
        parser.add_argument('--dim_hidden', type=int, default=128)
        parser.add_argument('--num_heads', type=int, default=4)
        parser.add_argument('--ln', type=bool, default=False)
        
        return parent_parser


    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if hparams['model_type'] == 'MultiTraceConv':
            self.backbone = MultiTraceConv(**hparams)
        elif hparams['model_type'] == 'SetTransformer':
            self.backbone = SetTransformer(**hparams)
        elif hparams['model_type'] == 'MultiTraceConvAttention':
            self.backbone = MultiTraceConvAttention(**hparams)
        elif hparams['model_type'] == 'SingleTraceConv':
            self.backbone = SingleTraceConv(**hparams)
        else:
            raise ValueError('Model type not recognized.')


    def forward(self, x):
        return self.backbone.forward(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

    def loss_fn(self, inputs, targets):
        return nn.functional.mse_loss(inputs, targets)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        pred = torch.squeeze(pred)
        y = torch.squeeze(y)
        loss = self.loss_fn(pred, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        pred = torch.squeeze(pred)
        y = torch.squeeze(y)
        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss)

    def __call__(self, traces,
            monotone_filter=False, monotone_filter_start=500, verbose=True,
            use_auto_batch_size=True, batch_size=-1, sort=True):
        ''' Run demixer over PSC trace batch and apply monotone decay filter.
        '''

        # define forward call for a single batch
        def _forward(traces):
            maxv = np.max(traces, axis=-1, keepdims=True)
            dem = self.forward(
                torch.tensor(
                    (traces/maxv), dtype=torch.float32, device=self.device
                )
            ).cpu().detach().numpy().squeeze() * maxv

            if monotone_filter:
                dem = cm.neural_waveform_demixing._monotone_decay_filter(dem, inplace=False,
                                                                        monotone_start=monotone_filter_start)
            return dem

        if verbose:
            print('Running photocurrent removal...', end='')
        t1 = time.time()

        # For multi-trace model, this will group traces of 
        # similar magnitudes to be in the same batch.
        if sort:
            idxs = np.argsort(np.sum(traces, axis=-1))
        else:
            idxs = np.arange(num_traces)
        
        # save this so that we can return estimates in the original (unsorted) order
        reverse_idxs = np.argsort(idxs)
        traces = traces[idxs]

        # if available, automatically break traces into the same size batches
        # used during training
        if use_auto_batch_size:
            batch_size = self.hparams['num_traces_per_experiment']
        if batch_size == -1:
            out = _forward(traces)
        else:
            out = np.zeros_like(traces)
            num_traces = traces.shape[0]
            start_idxs = np.arange(0, num_traces, batch_size)
            for idx in start_idxs:

                # stop instead of running on incomplete batch
                if idx+batch_size >= num_traces:
                    break
                out[idx:idx+batch_size] = _forward(traces[idx:idx+batch_size])
            
            # Run forward on the last batch, in case the number of traces is not
            # divisible by the batch size
            out[-batch_size:] = _forward(traces[-batch_size:])


        t2 = time.time()
        if verbose:
            print('complete (elapsed time %.2fs, device=%s).' %
                  (t2 - t1, self.device))

        return out[reverse_idxs]

    def generate_training_data(self,
                               args):
        pc_shape_params = dict(
            O_inf_min=args.O_inf_min,
            O_inf_max=args.O_inf_max,
            R_inf_min=args.R_inf_min,
            R_inf_max=args.R_inf_max,
            tau_o_min=args.tau_o_min,
            tau_o_max=args.tau_o_max,
            tau_r_min=args.tau_r_min,
            tau_r_max=args.tau_r_max,
        )
        key = jrand.PRNGKey(0)
        train_key, test_key = jrand.split(key, num=2)
        self.train_expts = photocurrent_sim.sample_photocurrent_expts_batch(
            train_key,
            args.num_train,
            args.num_traces_per_expt,
            trial_dur=900,
            pc_scale_range=(args.pc_scale_min, args.pc_scale_max),
            gp_scale_range=(args.gp_scale_min, args.gp_scale_max),
            iid_noise_scale_range=(
                args.iid_noise_scale_min, args.iid_noise_scale_max),
            min_pc_fraction=args.min_pc_fraction,
            max_pc_fraction=args.max_pc_fraction,
            gp_lengthscale=args.gp_lengthscale,
            pc_shape_params=pc_shape_params,
            add_target_gp=args.add_target_gp,
            target_gp_lengthscale=args.target_gp_lengthscale,
            target_gp_scale=args.target_gp_scale,
            linear_onset_frac=args.linear_onset_frac,
        )

        self.test_expts = photocurrent_sim.sample_photocurrent_expts_batch(
            test_key,
            args.num_test,
            args.num_traces_per_expt,
            trial_dur=900,
            pc_scale_range=(args.pc_scale_min, args.pc_scale_max),
            gp_scale_range=(args.gp_scale_min, args.gp_scale_max),
            iid_noise_scale_range=(
                args.iid_noise_scale_min, args.iid_noise_scale_max),
            min_pc_fraction=args.min_pc_fraction,
            max_pc_fraction=args.max_pc_fraction,
            gp_lengthscale=args.gp_lengthscale,
            pc_shape_params=pc_shape_params,
            add_target_gp=args.add_target_gp,
            target_gp_lengthscale=args.target_gp_lengthscale,
            target_gp_scale=args.target_gp_scale,
            linear_onset_frac=args.linear_onset_frac,
        )


class PhotocurrentData(torch.utils.data.Dataset):
    ''' Torch training dataset
    '''

    def __init__(self, expts):
        super(PhotocurrentData).__init__()
        self.N = expts[0].shape[0]
        self.obs = [np.array(x, dtype=np.float32) for x in expts[0]]
        self.targets = [np.array(x, dtype=np.float32) for x in expts[1]]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        these_obs = self.obs[idx]
        these_targets = self.targets[idx]
        maxv = np.max(these_obs, axis=-1, keepdims=True) + 1e-3
        maxv = np.abs(maxv) # traces not have negative max values

        these_obs /= maxv
        these_targets /= maxv

        return (these_obs, these_targets)
        
class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = float(v)
        setattr(namespace, self.dest, my_dict)


def parse_args(argseq):
    parser = ArgumentParser()

    # Add program level args. The arguments for PSC generation are passed as a separate dictionary
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--data_save_path', type=str, default="")
    parser.add_argument('--data_load_path', type=str, default="")
    # parser.add_argument("--psc_generation_kwargs", dest="psc_generation_kwargs", action=StoreDictKeyPair,
    #                     default=dict(gp_scale=0.045, delta_lower=160,
    #                                  delta_upper=400, next_delta_lower=400, next_delta_upper=899,
    #                                  prev_delta_upper=150, tau_diff_lower=60,
    #                                  tau_diff_upper=120, tau_r_lower=10,
    #                                  tau_r_upper=40, noise_std_lower=0.001,
    #                                  noise_std_upper=0.02, gp_lengthscale=45, sigma=30)
    #                     )

    parser = Subtractr.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(argseq)
    return args


if __name__ == "__main__":

    args = parse_args(sys.argv[1:])
    # Create subtractr and gen data
    subtractr = Subtractr(**vars(args)).float()

    # seed everything
    pl.seed_everything(0)

    if args.data_load_path != "":
        dat = np.load(args.data_load_path, allow_pickle=True)
        subtractr.train_expts = [(x[0], x[1]) for x in dat['train_expts']]
        subtractr.test_expts = [(x[0], x[1]) for x in dat['test_expts']]
    else:
        subtractr.generate_training_data(args)

    if args.data_save_path != "":
        np.savez(args.data_save_path, train_expts=subtractr.train_expts,
                 test_expts=subtractr.test_expts)

    train_dset = PhotocurrentData(expts=subtractr.train_expts)
    test_dset = PhotocurrentData(expts=subtractr.test_expts)
    train_dataloader = DataLoader(train_dset,
                                  pin_memory=True,
                                  batch_size=args.batch_size,
                                  num_workers=0)
    test_dataloader = DataLoader(test_dset,
                                 pin_memory=True,
                                 batch_size=args.batch_size,
                                 num_workers=0)

    # Run torch update loops
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(subtractr, train_dataloader, test_dataloader)
