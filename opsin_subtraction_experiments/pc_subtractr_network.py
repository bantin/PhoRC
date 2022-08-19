import tqdm
import numpy as np
import circuitmap as cm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import time
import math
import sys

import argparse
from argparse import ArgumentParser
import photocurrent_sim
import jax
import jax.random as jrand
jax.config.update('jax_platform_name', 'cpu')


class Subtractr(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Subtractr")
        parser.add_argument('--num_train', type=int, default=1000)
        parser.add_argument('--num_test', type=int, default=100)
        parser.add_argument('--trial_dur', type=int, default=900)
        parser.add_argument('--num_traces_per_expt', type=int, default=200)
        parser.add_argument('--photocurrent_scale_min',
                            type=float, default=0.01)
        parser.add_argument('--photocurrent_scale_max',
                            type=float, default=1.1)
        parser.add_argument('--pc_scale_min', type=float, default=0.05)
        parser.add_argument('--pc_scale_max', type=float, default=10.0)
        parser.add_argument('--gp_scale_min', type=float, default=0.01)
        parser.add_argument('--gp_scale_max', type=float, default=0.2)
        parser.add_argument('--iid_noise_scale_min', type=float, default=0.01)
        parser.add_argument('--iid_noise_scale_max', type=float, default=0.1)
        return parent_parser

    def __init__(self, args=None):
        super(Subtractr, self).__init__()

        # save hyperparms stored in args, if present
        self.save_hyperparameters()

        # U-NET for creating temporal waveform
        self.dblock1 = DownsamplingBlock(1, 16, 32, 2)
        self.dblock2 = DownsamplingBlock(16, 16, 32, 1)
        self.dblock3 = DownsamplingBlock(16, 32, 16, 1)
        self.dblock4 = DownsamplingBlock(32, 64, 16, 1)
        self.ublock1 = UpsamplingBlock(64, 32, 16, 1)
        self.ublock2 = UpsamplingBlock(32, 16, 16, 1)
        self.ublock3 = UpsamplingBlock(16, 16, 32, 1)
        self.ublock4 = UpsamplingBlock(16, 4, 32, 2)
        self.conv = ConvolutionBlock(4, 1, 256, 255, 1, 2)

        # Encoder for creating scalar multiplier
        self.scalar_block1 = DownsamplingBlock(2, 16, 32, 2)
        self.scalar_block2 = DownsamplingBlock(16, 16, 32, 2)
        self.scalar_block3 = DownsamplingBlock(16, 32, 32, 1)
        self.scalar_block4 = DownsamplingBlock(32, 1, 16, 1)


    def forward(self, x):
        # import pdb; pdb.set_trace()

        x = torch.squeeze(x)[:, None, :]  # batch x channel x time
        N, _, T = x.shape

        # Encoding and decoding for temporal waveform network
        enc1 = self.dblock1(x)
        enc2 = self.dblock2(enc1)
        enc3 = self.dblock3(enc2)
        enc4 = self.dblock4(enc3)

        # sum over batch dimension
        enc4 = torch.sum(enc4, dim=0, keepdim=True)
        dec1 = self.ublock1(enc4, interp_size=enc3.shape[-1])
        dec2 = self.ublock2(dec1, interp_size=enc2.shape[-1])
        dec3 = self.ublock3(dec2, interp_size=enc1.shape[-1])
        dec4 = self.ublock4(dec3, interp_size=x.shape[-1])

        # Final conv layer
        waveform = self.conv(dec4)
        waveform_broadcast = torch.broadcast_to(waveform, (N, 1, T))

        # Use encoder and waveform to create scalar for each input trace
        waveform_and_trace = torch.cat((waveform_broadcast, x), dim=1) # concat along channels
        senc1 = self.scalar_block1(waveform_and_trace)
        senc2 = self.scalar_block2(senc1)
        senc3 = self.scalar_block3(senc2)
        senc4 = self.scalar_block4(senc3)

        # senc4 should have dimensions N x 1 x T_small
        multipliers = torch.nn.functional.relu(torch.sum(senc4, dim=-1))

        # form rank-1 output
        return torch.outer(torch.squeeze(multipliers), torch.squeeze(waveform))



        

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)

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

    def run(self, traces, monotone_filter=False, monotone_filter_start=500, verbose=True, normalize=False):
        ''' Run demixer over PSC trace batch and apply monotone decay filter.
        '''

        if verbose:
            print('Running photocurrent removal...', end='')
        t1 = time.time()

        # Here, we normalize over a whole batch of traces, so that the largest amplitude is 1.
        # Note that this is different from NWD, where we scale each trace.
        if normalize:
            tmax = np.max(traces)
        else:
            tmax = 1.0
        dem = self.forward(
            torch.tensor(
                (traces/tmax), dtype=torch.float32, device=self.device
            )
        ).cpu().detach().numpy().squeeze() * tmax

        if monotone_filter:
            dem = cm.neural_waveform_demixing._monotone_decay_filter(dem, inplace=False,
                                                                     monotone_start=monotone_filter_start)

        t2 = time.time()
        if verbose:
            print('complete (elapsed time %.2fs, device=%s).' %
                  (t2 - t1, self.device))

        return dem

    def generate_training_data(self,
                               args):

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
        )

    # def train(self, epochs=1000, batch_size=64, learning_rate=1e-2, data_path=None, save_every=1,
    #           save_path=None, num_workers=2, pin_memory=True, num_gpus=1):
    #     ''' Run pytorch training loop.
    #     '''

    #     # print('CUDA device available: ', torch.cuda.is_available())
    #     # print('CUDA device: ', torch.cuda.get_device_name())

    #     if data_path is not None:
    #         raise NotImplementedError
    #     else:
    #         print('Attempting to load data from self object... ', end='')
    #         train_data = PhotocurrentData(start=0, end=len(
    #             self.train_expts), expts=self.train_expts)
    #         test_data = PhotocurrentData(start=0, end=len(
    #             self.test_expts), expts=self.test_expts)
    #         print('found.')

    #     train_dataloader = DataLoader(train_data,
    #                                   pin_memory=pin_memory, num_workers=num_workers)
    #     test_dataloader = DataLoader(test_data,
    #                                  pin_memory=pin_memory, num_workers=num_workers)

    #     # Run torch update loops
    #     print('Initiating neural net training...')
    #     t_start = time.time()
    #     self.trainer = pl.Trainer(gpus=num_gpus, max_epochs=epochs, precision=64, )
    #     self.trainer.fit(self.demixer, train_dataloader, test_dataloader)
    #     t_stop = time.time()


class PhotocurrentData(torch.utils.data.IterableDataset):
    ''' Torch training dataset
    '''

    def __init__(self, expts, min_traces_subsample=10):
        super(PhotocurrentData).__init__()
        self.obs = iter([np.array(x, dtype=np.float32) for x in expts[0]])
        self.targets = iter([np.array(x, dtype=np.float32) for x in expts[1]])
        self.min_traces_subsample = min_traces_subsample
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        these_obs = next(self.obs)
        these_targets = next(self.targets)

        num_traces = these_obs.shape[0]
        num_traces_to_keep = np.random.randint(
            low=self.min_traces_subsample, high=num_traces)
        idxs_to_keep = np.random.randint(
            low=0, high=num_traces, size=num_traces_to_keep)

        # normalize by the max, as we'll do at test time
        these_obs = these_obs[idxs_to_keep, :]
        these_obs /= (np.max(these_obs) + 1e-3)

        these_targets = these_targets[idxs_to_keep, :]
        these_targets /= (np.max(these_targets) + 1e-3)
        return (these_obs, these_targets)


class DownsamplingBlock(nn.Module):
    ''' DownsamplingBlock
    '''

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DownsamplingBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              dilation=dilation)
        self.decimate = nn.AvgPool1d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.relu(self.bn(self.conv(self.decimate(x))))


class UpsamplingBlock(nn.Module):
    ''' UpsamplingBlock
    '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, interpolation_mode='linear'):
        super(UpsamplingBlock, self).__init__()

        self.deconv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        self.interpolation_mode = interpolation_mode

    def forward(self, x, skip=None, interp_size=None):
        if skip is not None:
            up = nn.functional.interpolate(self.relu(self.bn(self.deconv(x))), size=skip.shape[-1],
                                           mode=self.interpolation_mode, align_corners=False)
            return torch.cat([up, skip], dim=1)
        else:
            return nn.functional.interpolate(self.relu(self.bn(self.deconv(x))), size=interp_size,
                                             mode=self.interpolation_mode, align_corners=False)


class ConvolutionBlock(nn.Module):
    ''' ConvolutionBlock
    '''

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation):
        super(ConvolutionBlock, self).__init__()

        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


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
    parser.add_argument('--batch_size', type=int, default=32)
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
    subtractr = Subtractr(args).float()

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
                                  pin_memory=True, num_workers=0)
    test_dataloader = DataLoader(test_dset,
                                 pin_memory=True, num_workers=0)

    # Run torch update loops
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(subtractr, train_dataloader, test_dataloader)
