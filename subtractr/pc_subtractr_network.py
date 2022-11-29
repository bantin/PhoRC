import numpy as np
# import circuitmap as cm

import torch
import torch.nn as nn
import pytorch_lightning as pl
import time
import sys
import os
import shutil
import atexit
import argparse
import jax
import jax.numpy as jnp
import jax.random as jrand
import glob
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import subtractr.photocurrent_sim as photocurrent_sim
import subtractr.backbones as backbones

from functools import partial
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from jax import vmap
from subtractr.photocurrent_sim import sample_photocurrent_experiment

# jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'platform'


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

        # units for these scales are nA
        parser.add_argument('--pc_scale_min', type=float, default=0.01)
        parser.add_argument('--pc_scale_max', type=float, default=2.0)
        parser.add_argument('--psc_scale_min', type=float, default=0.01)
        parser.add_argument('--psc_scale_max', type=float, default=0.5) 

        parser.add_argument('--gp_scale_min', type=float, default=0.01)
        parser.add_argument('--gp_scale_max', type=float, default=0.045)
        parser.add_argument('--iid_noise_std_min', type=float, default=0.001)
        parser.add_argument('--iid_noise_std_max', type=float, default=0.02)
        parser.add_argument('--min_pc_fraction', type=float, default=0.5)
        parser.add_argument('--max_pc_fraction', type=float, default=1.0)
        parser.add_argument('--gp_lengthscale', type=float, default=45.0)

        # whether we use the LS solver at the end of forward pass
        parser.add_argument('--use_ls_solve', action='store_true')
        parser.add_argument('--no_use_ls_solve', dest='use_ls_solve', action='store_false')
        parser.set_defaults(use_ls_solve=False)

        # define how we normalize the input data
        parser.add_argument('--normalize_type', default='max')
        
        # whether we add a gp to the target waveforms
        parser.add_argument('--add_target_gp', action='store_true')
        parser.add_argument('--no_add_target_gp', dest='add_target_gp', action='store_false')
        parser.set_defaults(add_target_gp=True)
        parser.add_argument('--target_gp_lengthscale', type=float, default=25)
        parser.add_argument('--target_gp_scale', type=float, default=0.01)

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

        # oneCycleLR
        parser.add_argument('--use_onecyclelr', action='store_true')
        parser.add_argument('--no_use_onecyclelr', dest='use_onecyclelr', action='store_false')
        parser.set_defaults(use_onecyclelr=False)
        parser.add_argument('--onecyclelr_max_lr', type=float, default=1e-1)
        parser.add_argument('--onecyclelr_div_factor', type=float, default=25)
        parser.add_argument('--onecyclelr_final_div_factor', type=float, default=1e4)
        
        return parent_parser


    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if hparams['model_type'] == 'MultiTraceConv':
            self.backbone = backbones.MultiTraceConv(**hparams)
        elif hparams['model_type'] == 'SetTransformer':
            self.backbone = backbones.SetTransformer(**hparams)
        elif hparams['model_type'] == 'MultiTraceConvAttention':
            self.backbone = backbones.MultiTraceConvAttention(**hparams)
        elif hparams['model_type'] == 'SingleTraceConv':
            self.backbone = backbones.SingleTraceConv(**hparams)
        elif hparams['model_type'] == 'DeepLowRank':
            self.backbone = backbones.DeepLowRank(**hparams)
        else:
            raise ValueError('Model type not recognized.')


    def forward(self, x):
        return self.backbone.forward(x)

    def configure_optimizers(self):

        optim = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        if args.use_onecyclelr:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim,
                total_steps=self.num_training_steps,
                max_lr=args.onecyclelr_max_lr,
                div_factor=args.onecyclelr_div_factor,
                final_div_factor=args.onecyclelr_final_div_factor,
            )
            return [optim], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optim

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
        return loss

    def __call__(self, traces,
            monotone_filter=False, monotone_filter_start=500, verbose=True,
            use_auto_batch_size=True, batch_size=-1, sort=True,
            max_norm_start_idx=100, max_norm_end_idx=700):
        ''' Run demixer over PSC trace batch and apply monotone decay filter.
        '''
        self.eval()
        # define forward call for a single batch
        def _forward(traces):
            normalize_type = self.hparams.get('normalize_type', 'l2')
            if normalize_type == 'l2':
                maxv = (np.linalg.norm(traces) / traces.shape[0])
            elif normalize_type == 'max':
                maxv = np.max(traces, axis=-1, keepdims=True)

            dem = self.forward(
                torch.tensor(
                    (traces/maxv)[None,:,:], dtype=torch.float32, device=self.device
                )
            ).cpu().detach().numpy().squeeze() * maxv

            if monotone_filter:
                filt_start=500
                filtered = photocurrent_sim.monotone_decay_filter(dem, monotone_start=filt_start)
                dem.at[:,filt_start:].set(filtered)
            return dem

        if verbose:
            print('Running photocurrent removal...', end='')
        t1 = time.time()

        # For multi-trace model, this will group traces of 
        # similar magnitudes to be in the same batch.
        if sort:
            idxs = np.argsort(np.linalg.norm(traces[:, 100:250], axis=-1))
        else:
            idxs = np.arange(num_traces)
        
        # save this so that we can return estimates in the original (unsorted) order
        reverse_idxs = np.argsort(idxs)
        traces = traces[idxs]

        # if available, automatically break traces into the same size batches
        # used during training
        if use_auto_batch_size:
            batch_size = self.hparams['num_traces_per_expt']
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

        psc_shape_params = dict(
            tau_r_lower = 10,
            tau_r_upper = 40,
            tau_diff_lower = 60,
            tau_diff_upper = 120,
            delta_lower=160,
            delta_upper=400,
            next_delta_lower=400,
            next_delta_upper=899,
            prev_delta_upper=150,
            amplitude_lower=args.psc_scale_min,
            amplitude_upper=args.psc_scale_max,
        )

        key = jrand.PRNGKey(0)
        keys = iter(jrand.split(key, num=2))

        sampler_func = vmap(partial(sample_photocurrent_experiment,
            num_traces=args.num_traces_per_expt, 
            onset_jitter_ms=args.onset_jitter_ms,
            onset_latency_ms=args.onset_latency_ms,
            pc_shape_params=pc_shape_params,
            psc_shape_params=psc_shape_params,
            min_pc_scale=args.pc_scale_min,
            max_pc_scale=args.pc_scale_max,
            min_pc_fraction=args.min_pc_fraction,
            max_pc_fraction=args.max_pc_fraction,
            add_target_gp=args.add_target_gp,
            target_gp_lengthscale=args.target_gp_lengthscale,
            target_gp_scale=args.target_gp_scale,
            linear_onset_frac=args.linear_onset_frac,
            msecs_per_sample=0.05,
            stim_start=5.0,
            tstart=-10.0,
            tend=47.0,
            time_zero_idx=200,
            normalize_type=args.normalize_type,
            iid_noise_std_min=args.iid_noise_std_min,
            iid_noise_std_max=args.iid_noise_std_max))

        # Since low-pass filters are not yet implemented in jax, we
        # do this as a postprocessing step
        def sample_and_postprocess(keys):
            obs, targets = sampler_func(keys)
            # obs = postprocess_photocurrent_experiment_batch(obs)
            return (obs, targets)

        train_keys = jrand.split(next(keys), args.num_train)
        test_keys = jrand.split(next(keys), args.num_test)

        if args.data_on_disk:

            # for SLURM jobs, make sure that two jobs running on 
            # the same node will not save train data to the same place
            folder_suffix = ''
            slurmid = os.getenv('SLURM_JOB_ID')
            if slurmid is not None:
                folder_suffix = slurmid

            # make train/test folders
            self.train_path = os.path.join(args.data_save_path, 'train' + folder_suffix + '/')
            self.test_path = os.path.join(args.data_save_path, 'test' + folder_suffix + '/')
            os.makedirs(self.train_path, exist_ok=True)
            os.makedirs(self.test_path, exist_ok=True)

            for (keyset, path, label) in zip(
                    [train_keys, test_keys],
                    [self.train_path, self.test_path],
                    ['train', 'test']):
                keyset_batched = jnp.split(keyset, jnp.arange(0, keyset.shape[0], args.batch_size)[1:])
                for batch_idx,  these_keys in enumerate(keyset_batched):
                    expts = sample_and_postprocess(these_keys) 
                    curr_path = os.path.join(path, '%s_batch%d' % (label, batch_idx))
                    np.save(curr_path, expts)

        else:
            self.train_expts = sample_and_postprocess(train_keys)
            self.test_expts = sample_and_postprocess(test_keys)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps > 0:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = int(np.ceil(self.hparams['num_train'] / self.hparams['batch_size']))
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_devices)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices

        return (batches // effective_accum) * self.trainer.max_epochs

class MemDataset(torch.utils.data.Dataset):
    ''' Torch training dataset
    '''

    def __init__(self, expts):
        super(MemDataset).__init__()
        self.N = expts[0].shape[0]
        self.obs = expts[0]
        self.targets = expts[1]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        these_obs = np.array(self.obs[idx], dtype=np.float32)
        these_targets = np.array(self.targets[idx], dtype=np.float32)

        return (these_obs, these_targets)


class DiskDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.files = glob.glob(path + '*.npy')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        (inputs, targets) = np.load(self.files[idx])
        return (np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32))
        
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

    # whether we put the datset on disk
    parser.add_argument('--data_on_disk', action='store_true')
    parser.add_argument('--no_data_on_disk', dest='data_on_disk', action='store_false')
    parser.set_defaults(data_on_disk=False)

    # whether we erase training/test data when training completes.
    parser.add_argument('--cleanup', action='store_true')
    parser.add_argument('--no_cleanup', dest='cleanup', action='store_false')
    parser.set_defaults(cleanup=True)

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
       raise NotImplementedError
    else:
        subtractr.generate_training_data(args)


    subtractr = subtractr.to(torch.float)


    if args.data_on_disk:
        train_dset = DiskDataset(subtractr.train_path)
        test_dset = DiskDataset(subtractr.test_path)
        effective_batch_size = None

        # free any gpu mem used by jax
        backend = jax.lib.xla_bridge.get_backend()
        for buf in backend.live_buffers():
            buf.delete()

        
    else:
        train_dset = MemDataset(subtractr.train_expts)
        test_dset = MemDataset(subtractr.test_expts)
        effective_batch_size = args.batch_size

    train_dataloader = DataLoader(train_dset,
                                pin_memory=True,
                                batch_size=effective_batch_size,
                                sampler=None,
                                shuffle=True,
                                num_workers=args.num_workers)

    test_dataloader = DataLoader(test_dset,
                                pin_memory=True,
                                batch_size=effective_batch_size,
                                sampler=None,
                                num_workers=args.num_workers)

    # configure checkpoint callbacks
    val_checkpoint_callback = ModelCheckpoint(save_top_k=1,
        monitor="val_loss", filename='{epoch}-{val_loss}_best_val')
    periodic_checkpoint_callback = ModelCheckpoint(
        every_n_epochs=10, monitor=None, save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Run torch update loops
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[val_checkpoint_callback,
        periodic_checkpoint_callback, lr_monitor])
    trainer.fit(subtractr, train_dataloader, test_dataloader)

    # cleanup
    if args.data_on_disk and args.cleanup:
        shutil.rmtree(subtractr.train_path)
        shutil.rmtree(subtractr.test_path)
