import numpy as np
import circuitmap as cm

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import time
import math

import tqdm

def _sample_gp(trial_dur=800, gp_lengthscale=25, gp_scale=0.01, n_samples=1):
    D = np.array([[i - j for i in range(trial_dur)] for j in range(trial_dur)])
    K = np.exp(-D**2/(2 * gp_lengthscale**2))
    mean = np.zeros(trial_dur)
    return gp_scale * np.random.multivariate_normal(mean, K, size=n_samples)


def _sample_photocurrent_params(stim_off_current_min=0.02,
                                stim_off_current_max=0.1,
                                tau_r_min=100,
                                tau_r_max=200,
                                tau_d_min=20,
                                tau_d_max=80):
    return dict(
        stim_off_current=np.random.uniform(
            low=stim_off_current_min, high=stim_off_current_max,),
        tau_r=np.random.uniform(low=tau_r_min, high=tau_r_max,),
        tau_d=np.random.uniform(low=tau_d_min, high=tau_d_max)
    )


def gen_photocurrent_waveform(stim_on=100, stim_off=200, trial_dur=900, stim_off_current=0.07, tau_r=130, tau_d=300):
    out = np.zeros(trial_dur)
    x = np.arange(trial_dur)

    # make linear rise from stim on to stim off
    out[stim_on:stim_off] = np.linspace(
        start=0, stop=stim_off_current, num=(stim_off - stim_on))

    # calculate offset for continuity
    delta = stim_on
    pc_shape = (tau_d * tau_r / (tau_d - tau_r)) * (np.exp(-(x -
                                                             delta)/tau_d) - np.exp(-(x - delta)/tau_r)) * (x >= delta)
    pc_shape /= np.max(pc_shape)
    pc_shape *= stim_off_current / pc_shape[stim_off]
    out[stim_off:] = pc_shape[stim_off:]

    # add GP to photocurrent shape for added variability
    gp = np.squeeze(_sample_gp(
        trial_dur=900, gp_scale=0.004, gp_lengthscale=100))

    out += gp
    out = np.maximum(0, out)
    out[0:stim_on] = 0

    # ensure waveform is monotonically decreasing after 400 frames
    out = np.squeeze(cm.neural_waveform_demixing._monotone_decay_filter(
        out[None, :], monotone_start=400, inplace=False))

    # convolve with gaussian to smooth the edges
    return gaussian_filter1d(out, sigma=10)


def gen_scaled_photocurrents(
            trial_dur=900,
            num_traces=100,
            photocurrent_scale_min=0.03,
            photocurrent_scale_max=1.5,
            photocurrent_fraction=0.3,
            stim_off_current_min=0.02,
            stim_off_current_max=0.2,
            tau_r_min=100,
            tau_r_max=200,
            tau_d_min=20,
            tau_d_max=80,
            per_trial_gp_scale=0.001,
            per_trial_gp_lengthscale=10,
                             ):
    out = np.zeros((num_traces, trial_dur))

    # Generate a single photocurrent template used across the whole experiment
    pc_params = _sample_photocurrent_params(
        stim_off_current_min=stim_off_current_min,
        stim_off_current_max=stim_off_current_max,
        tau_r_min=tau_r_min,
        tau_r_max=tau_r_max,
        tau_d_min=tau_d_min,
        tau_d_max=tau_d_max,
    )
    pc_template = gen_photocurrent_waveform(**pc_params)
    pc_template /= np.max(pc_template)

    # on each trial, the _true_ photocurrent is corrupted slightly by GP noise,
    # which is constrained to be decreasing after 500 frames
    gp_noise = _sample_gp(trial_dur=trial_dur,
        gp_scale=per_trial_gp_scale, gp_lengthscale=per_trial_gp_lengthscale,
        n_samples=num_traces)
    cm.neural_waveform_demixing._monotone_decay_filter(gp_noise,)
    out = gp_noise

    # generate random scaling of the photocurrent for each trace,
    # then set some photocurrents to zero
    scales = np.random.uniform(
        low=photocurrent_scale_min, high=photocurrent_scale_max, size=(num_traces))
    scales *= np.random.rand(num_traces) >= photocurrent_fraction
    out += scales[:, None] * \
        np.broadcast_to(pc_template, shape=(num_traces, trial_dur))

    return out


def gen_photocurrent_data(trial_dur=900,
                          num_expts=1000,
                          min_traces_per_expt=100,
                          max_traces_per_expt=1000,
                          photocurrent_scale_min=0.05,
                          photocurrent_scale_max=0.8,
                          photocurrent_fraction=0.3,
                          stim_off_current_min=0.02,
                          stim_off_current_max=0.1,
                          tau_r_min=100,
                          tau_r_max=200,
                          tau_d_min=20,
                          tau_d_max=80,
                          psc_generation_kwargs=None,
                        ):
    # Generate length (in traces) for each experiment
    exp_lengths = np.random.randint(low=min_traces_per_expt,
        high=max_traces_per_expt, size=num_expts)
    inputs = np.zeros((np.sum(exp_lengths), trial_dur))
    targets = np.zeros((np.sum(exp_lengths), trial_dur))

    # generate all psc traces from neural demixer
    if psc_generation_kwargs is None:
        psc_generation_kwargs = dict()
    demixer = cm.NeuralDemixer()
    demixer.generate_training_data(
        size=np.sum(exp_lengths), training_fraction=1.0, **psc_generation_kwargs)
    pscs, _ = demixer.training_data

    
    # Indexing here is convoluted: we want chunks of traces (rows of inputs, targets)
    # to represent data from the same "experiment". We create the exp_idxs list 
    # which is used to index into inputs/targets and denotes the boundaries of where
    # each experiment begins and ends. This is _much_ faster than appending to a list 
    # inside the loop.
    exp_idxs = [0, *[n for n in np.cumsum(exp_lengths)]]

    # For each experiment, we generate a group of traces that share a common photocurrent component.
    # Some random fraction of these will _not_ have photocurrent present.
    for i in tqdm.trange(num_expts):

        start_idx = exp_idxs[i]
        end_idx = exp_idxs[i+1]

        scaled_pcs = gen_scaled_photocurrents(num_traces=end_idx - start_idx,
            photocurrent_scale_min=photocurrent_scale_min,
            photocurrent_scale_max=photocurrent_scale_max,
            photocurrent_fraction=photocurrent_fraction,
            stim_off_current_min=stim_off_current_min,
            stim_off_current_max=stim_off_current_max,
            tau_r_min=tau_r_min,
            tau_r_max=tau_r_max,
            tau_d_min=tau_d_min,
            tau_d_max=tau_d_max,
        )
        inputs[start_idx:end_idx] = scaled_pcs + pscs[start_idx:end_idx]
        targets[start_idx:end_idx] = scaled_pcs

    # The listcomp here should be fast?
    expts = [(inputs[start_idx:end_idx], targets[start_idx:end_idx])
        for (start_idx, end_idx) in zip(exp_idxs[0:-1], exp_idxs[1:])]
    return expts


class Subtractr():

    def __init__(self, path=None, eval_mode=True, device=None):
        # Set device dynamically
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load or initialise demixer object
        if path is not None:
            self.demixer = NWDUNet().load_from_checkpoint(path)
            if eval_mode:
                self.demixer.eval()
        else:
            self.demixer = NWDUNet()

        # Move demixer to device
        self.demixer = self.demixer.to(self.device)

    def __call__(self, traces, monotone_filter_start=500, monotone_filter_inplace=True, verbose=True):
        ''' Run demixer over PSC trace batch and apply monotone decay filter.
        '''

        if verbose:
            print('Demixing PSC traces... ', end='')
        t1 = time.time()

        tmax = np.max(traces, axis=1)[:, None]
        dem = self.demixer(
            torch.Tensor((traces/tmax).copy()
                         [:, None, :]).to(device=self.device)
        ).cpu().detach().numpy().squeeze() * tmax

        # dem = _monotone_decay_filter(dem, inplace=monotone_filter_inplace,
        # 	monotone_start=monotone_filter_start)

        t2 = time.time()
        if verbose:
            print('complete (elapsed time %.2fs, device=%s).' %
                  (t2 - t1, self.device))

        return dem

    def generate_training_data(self,
            num_train,
            num_test,
            trial_dur=900,
            min_traces_per_expt=100,
            max_traces_per_expt=1000,
            photocurrent_scale_min=0.05,
            photocurrent_scale_max=0.8,
            photocurrent_fraction=0.3,
            stim_off_current_min=0.02,
            stim_off_current_max=0.1,
            tau_r_min=100,
            tau_r_max=200,
            tau_d_min=20,
            tau_d_max=80,
            psc_generation_kwargs=None):
        
        
        self.train_expts = gen_photocurrent_data(
            trial_dur=trial_dur,
            num_expts=num_train,
            min_traces_per_expt=min_traces_per_expt,
            max_traces_per_expt=max_traces_per_expt,
            photocurrent_scale_min=photocurrent_scale_min,
            photocurrent_scale_max=photocurrent_scale_max,
            photocurrent_fraction=photocurrent_fraction,
            stim_off_current_min=stim_off_current_min,
            stim_off_current_max=stim_off_current_max,
            tau_r_min=tau_r_min,
            tau_r_max=tau_r_max,
            tau_d_min=tau_d_min,
            tau_d_max=tau_d_max,
            psc_generation_kwargs=psc_generation_kwargs,
        )
        self.test_expts = gen_photocurrent_data(
            trial_dur=trial_dur,
            num_expts=num_test,
            min_traces_per_expt=min_traces_per_expt,
            max_traces_per_expt=max_traces_per_expt,
            photocurrent_scale_min=photocurrent_scale_min,
            photocurrent_scale_max=photocurrent_scale_max,
            photocurrent_fraction=photocurrent_fraction,
            stim_off_current_min=stim_off_current_min,
            stim_off_current_max=stim_off_current_max,
            tau_r_min=tau_r_min,
            tau_r_max=tau_r_max,
            tau_d_min=tau_d_min,
            tau_d_max=tau_d_max,
            psc_generation_kwargs=psc_generation_kwargs,
        )

    def train(self, epochs=1000, batch_size=64, learning_rate=1e-2, data_path=None, save_every=1,
              save_path=None, num_workers=2, pin_memory=True, num_gpus=1):
        ''' Run pytorch training loop.
        '''

        # print('CUDA device available: ', torch.cuda.is_available())
        # print('CUDA device: ', torch.cuda.get_device_name())

        if data_path is not None:
            raise NotImplementedError
        else:
            print('Attempting to load data from self object... ', end='')
            train_data = PhotocurrentData(start=0, end=len(
                self.train_expts), expts=self.train_expts)
            test_data = PhotocurrentData(start=0, end=len(
                self.test_expts), expts=self.test_expts)
            print('found.')

        train_dataloader = DataLoader(train_data,
                                      pin_memory=pin_memory, num_workers=num_workers)
        test_dataloader = DataLoader(test_data,
                                     pin_memory=pin_memory, num_workers=num_workers)

        # Run torch update loops
        print('Initiating neural net training...')
        t_start = time.time()
        self.trainer = pl.Trainer(gpus=num_gpus, max_epochs=epochs, precision=64)
        self.trainer.fit(self.demixer, train_dataloader, test_dataloader)
        t_stop = time.time()

        print("Training complete. Elapsed time: %.2f min." %
              ((t_stop-t_start)/60))


class PhotocurrentData(torch.utils.data.IterableDataset):
    ''' Torch training dataset
    '''

    def __init__(self, start, end, expts):
        super(PhotocurrentData).__init__()
        assert end > start, "end must be greater than start"
        self.end = end
        self.start = start
        self.expts = expts

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(self.expts[iter_start:iter_end])


class StackedDenoisingNetwork(nn.Module):
    ''' Denoising neural network consisting of multiple layers of long 1d convolutions.
    '''

    def __init__(self, n_layers=3, kernel_size=99, padding=49, channels=[16, 8, 1], stride=1):
        super(StackedDenoisingNetwork, self).__init__()
        assert n_layers >= 2, 'Neural network must have at least one input layer and one output layer.'
        assert channels[-1] == 1, 'Output layer must have exactly one output channel'

        layers = [nn.Conv1d(in_channels=1, out_channels=channels[0], kernel_size=kernel_size,
                            stride=stride, padding=padding)]
        layers.append(nn.ReLU())

        for l in range(1, n_layers):
            layers.append(nn.Conv1d(in_channels=channels[l - 1], out_channels=channels[l],
                                    kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


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


class NWDUNet(pl.LightningModule):
    ''' Neural waveform demixing U-Net
    '''

    def __init__(self):
        super(NWDUNet, self).__init__()
        self.feature_encoder = torch.nn.ModuleList([
            DownsamplingBlock(1, 16, 32, 2),
            DownsamplingBlock(16, 16, 32, 1),
            DownsamplingBlock(16, 32, 16, 1),
            DownsamplingBlock(32, 32, 16, 1)
        ])

        self.context_encoder = torch.nn.ModuleList([
            DownsamplingBlock(1, 16, 32, 2),
            DownsamplingBlock(16, 16, 32, 1),
            DownsamplingBlock(16, 32, 16, 1),
            DownsamplingBlock(32, 32, 16, 1)
        ])

        self.ublock1 = UpsamplingBlock(64, 48, 16, 1)
        self.ublock2 = UpsamplingBlock(48 + 32, 32, 16, 1)
        self.ublock3 = UpsamplingBlock(32 + 16, 16, 32, 1)
        self.ublock4 = UpsamplingBlock(16 + 16, 4, 32, 2)

        self.conv = ConvolutionBlock(4, 1, 256, 255, 1, 2)

    def forward(self, x):

        x = torch.squeeze(x)[:,None,:] # batch x channel x time

        # make feature vector which is aggregate over entire batch
        feats = torch.clone(x)
        for l in self.feature_encoder:
            feats = l(feats)
        dims = feats.shape
        feats = torch.mean(feats, dim=0, keepdim=True) # average over entire batch
        feats = torch.broadcast_to(feats, dims) #shape of feats now matches shape of inputs

        # Make context embedding, saving outputs to use as skip connections
        context = torch.clone(x)
        skip_inputs = []
        context_sizes = np.zeros(len(self.context_encoder), dtype=int)
        for l in self.context_encoder:
            skip_inputs.append(context)
            context = l(context) 

        # Concatenate context and features along channels dimension
        context_plus_features = torch.concat((context, feats), dim=1) # channels dimension

        # Decoding
        # NOTE: could add skip conns here
        dec1 = self.ublock1(context_plus_features, skip=skip_inputs[3])
        dec2 = self.ublock2(dec1, skip=skip_inputs[2])
        dec3 = self.ublock3(dec2, skip=skip_inputs[1])
        dec4 = self.ublock4(dec3, interp_size=x.shape[-1])

        # Final conv layer
        out = self.conv(dec4)

        return out

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


if __name__ == "__main__":
    from pc_subtractr_network import Subtractr
    torch.set_default_dtype(torch.float64)
    pc_subtractr = Subtractr()
    pc_subtractr.generate_training_data(
        num_train=2, num_test=1)
    pc_subtractr.train(num_gpus=0)
