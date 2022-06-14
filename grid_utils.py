import numpy as np
import matplotlib as mpl

from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path


from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt



def sequential_map(vals):
    uniques = np.unique(vals)
    return {x: i for i,x in enumerate(uniques)}

def sequential_map_nd(vals, axis):
    uniques = np.unique(vals, axis=axis)
    return {tuple(x): i for i,x in enumerate(uniques)}

def make_stim_matrix(I, L):
    # expect L to have shape trials x spots x 3
    num_trials, num_spots, _ = L.shape

    # Now extract unique x, y, z
    L_flat  = L.reshape(-1, 3)
    unique_locs = np.unique(L_flat, axis=0)
    loc_map = sequential_map_nd(L_flat, axis=0)
    num_neurons = unique_locs.shape[0]

    stim = np.zeros((num_neurons, num_trials))

    # for each trial, convert the tuple x,y,z to a single liner index (from 0 to num_locs)
    # then set that entry of the stim matrix
    for trial in range(num_trials):
        for spot in range(num_spots):
            neuron_idx = loc_map[tuple(L[trial, spot,:])]
            stim[neuron_idx, trial] = I[trial]

    return stim, loc_map



def make_psc_tensor(psc, I, L):
    '''
    Stack all observations into a PSC tensor of shape powers x height x width x trials x time.
    
    If not all pixels were hit the same number of times, pad the array with nan.
    '''
    num_trials = L.shape[0]
    powers = np.unique(I)
    num_powers = len(powers)
    timesteps = psc.shape[-1]
    
    # First, we need to compute the maximum number of times a pixel was stimmed.
    # If a pixel was stimmed 10x at 30mW and 10x at 50mW, we want to count these separately,
    # so we stack powers and locations to find stims at each unique power.
    unique_locs, counts = np.unique( np.c_[L,I], axis=0, return_counts=True)
    max_stims = np.max(counts)
    
    # Now extract unique x, y, z
    xs = np.unique(L[:,0])
    ys = np.unique(L[:,1])
    zs = np.unique(L[:,2])
    
    # create maps to map from location in real coordinates to location in index coordinates
    x_map = sequential_map(xs)
    y_map = sequential_map(ys)
    z_map = sequential_map(zs)
    p_map = sequential_map(powers)
    
    # Create array and fill with nan
    psc_tensor = np.full(
        (num_powers, len(xs), len(ys), len(zs), max_stims, timesteps),
        fill_value=np.nan)
    stim_inds = np.zeros((num_powers, len(xs), len(ys), len(zs)), dtype=int)

    # create empty stim matrix
    N = np.unique(L, axis=0).shape[0]
    stim_mat = np.zeros((N, num_trials))
    
    dims = (len(xs), len(ys), len(zs))
    for trial in range(num_trials):

        # pack psc traces into tensor by location and power
        powerloc_idx = ( p_map[I[trial]],
                x_map[L[trial, 0]],
                y_map[L[trial, 1]],
                z_map[L[trial, 2]],
        )
        
        stim_idx = stim_inds[powerloc_idx]
        combined_idx = (*powerloc_idx, stim_idx)
        psc_tensor[combined_idx] = psc[trial]
        
        # increment number of stims for a given location at a given power
        stim_inds[powerloc_idx] += 1

        # Fill in stim matrix
        loc_idx = (x_map[L[trial,0]], y_map[L[trial, 1]], z_map[L[trial,2]])
        pixel_idx = np.ravel_multi_index(loc_idx, dims)
        stim_mat[pixel_idx, trial] = I[trial]
        
    return psc_tensor

def get_max_hits(I, stim):
    powers = np.unique(I)
    num_powers = len(powers)
    max = 0
    for power in powers:
        curr_max = np.max(np.sum(stim == power, axis=-1), axis=0)
        if curr_max > max:
            max = curr_max
    return max


def make_psc_tensor_multispot(psc, I, L, stim):
    '''
    Stack all observations into a PSC tensor of shape powers x height x width x trials x time.
    
    If not all pixels were hit the same number of times, pad the array with nan.

    arguments:
        L: array of size (num_stims_per_round x num_spots x 3). Here a "round" is running through all holos.
        I: array of size (num_stims_per_round*num_rounds)
        stim: array of size (num_pixels x num_stims_per_round*num_rounds)
    '''
    num_trials = I.shape[0]
    powers = np.unique(I)
    num_powers = len(powers)
    timesteps = psc.shape[-1]
    num_spots = L.shape[1]
    
    # First, we need to compute the maximum number of times a pixel was stimmed.
    # If a pixel was stimmed 10x at 30mW and 10x at 50mW, we want to count these separately,
    # so we stack powers and locations to find stims at each unique power.
    max_stims = get_max_hits(I, stim)
    
    # Now extract unique x, y, z
    L_unrolled = L.reshape(-1, 3)
    xs = np.unique(L_unrolled[:,0])
    ys = np.unique(L_unrolled[:,1])
    zs = np.unique(L_unrolled[:,2])
    
    # create maps to map from location in real coordinates to location in index coordinates
    x_map = sequential_map(xs)
    y_map = sequential_map(ys)
    z_map = sequential_map(zs)
    p_map = sequential_map(powers)
    
    # Create array and fill with nan
    psc_tensor = np.zeros((num_powers, len(xs), len(ys), len(zs), max_stims, timesteps)) + np.nan
    stim_inds = np.zeros((num_powers, len(xs), len(ys), len(zs)), dtype=int)
    
    for trial in range(num_trials):
        for spot in range(num_spots):
            powerloc_idx = ( p_map[I[trial]],
                    x_map[L[trial, spot, 0]],
                    y_map[L[trial, spot, 1]],
                    z_map[L[trial, spot, 2]],
            )
            
            stim_idx = stim_inds[powerloc_idx]
            combined_idx = (*powerloc_idx, stim_idx)
            psc_tensor[combined_idx] = psc[trial]
            
            # increment number of stims for a given location at a given power
            stim_inds[powerloc_idx] += 1
        
    return psc_tensor    

def stack_observations_in_grid(y, I, L):
    num_trials = L.shape[0]
    power_levels = np.unique(I)

    xs = np.unique(L[:,0])
    ys = np.unique(L[:,1])
    zs = np.unique(L[:,2])
    
    # create arrays to store receptive fields
    x_map = sequential_map(xs)
    y_map = sequential_map(ys)
    z_map = sequential_map(zs)
    p_map = sequential_map(power_levels)
    
    obs = np.empty((len(power_levels), len(xs), len(ys), len(zs)), dtype=object)
    # initialize everything to the empty list.
    # unfortunately there's no easy way to do it.
    for pidx in range(len(power_levels)):
        for xidx in range(len(xs)):
            for yidx in range(len(ys)):
                for zidx in range(len(zs)):
                    obs[pidx, xidx, yidx, zidx] = []
            
    num_stims = np.zeros_like(obs, dtype=float)
    
    for trial in range(num_trials):
        idx = ( p_map[I[trial]],
                x_map[L[trial, 0]],
                y_map[L[trial, 1]],
                z_map[L[trial, 2]],
        )
        
        obs[idx].append(y[trial])
        num_stims[idx] += 1

    return obs, num_stims



def make_suff_stats(y,I,L):
    obs, num_stims = stack_observations_in_grid(y, I, L) 
    obs_mean = np.vectorize(lambda x: 0 if x == [] else np.mean(x))(obs)
    obs_var = np.vectorize(lambda x: 0 if x == [] else np.var(x))(obs)

        
    return obs_mean, num_stims, obs_var


def plot_multi_means(fig, mean_maps, depth_idxs,
                     zs=None, zlabels=None, powers=None, map_names=None, cmaps='viridis',
                     vranges=None, cbar_labels=None):
    
    # allow option to pass separate cmaps for each grid plot
    if not isinstance(cmaps, list):
        cmaps = len(mean_maps) * [cmaps]
        
    for mean_idx, mean_map, cmap in zip(
        range(len(mean_maps)), mean_maps, cmaps):
        
        num_powers, _, _, num_planes = mean_map.shape
        num_planes_to_plot = len(depth_idxs)
        assert num_planes_to_plot <= num_planes
            
        # Create a new grid for each mean map                
        subplot_args = int("1" + str(len(mean_maps)) + str(mean_idx + 1))
        ax_curr = plt.subplot(subplot_args)
        
        if powers is not None and map_names is not None:
            ax_curr.set_title(map_names[mean_idx], y=1.08)
            
        plt.axis('off')
        
        grid = ImageGrid(fig, subplot_args,  # similar to subplot(111)
                         nrows_ncols=(num_planes_to_plot, num_powers),  # creates 2x2 grid of axes
                         axes_pad=0.05,  # pad between axes in inch.
                         cbar_mode='single',
                         cbar_pad=0.2
                         )
        if vranges is not None:
            min_val, max_val = vranges[mean_idx]
        else:
            min_val = np.nanmin(mean_map)
            max_val = np.nanmax(mean_map)
        
        for j, ax in enumerate(grid):
            row = j // num_powers
            col = j % num_powers
            ax.set_xticks([])
            ax.set_yticks([])
#             ax.set_frame_on(False)

            # optionally add labels
            if (zlabels or zs) and col == 0:
                if zs:
                    ax.set_ylabel('%d ' % zs[depth_idxs[row]] + r'$\mu m $' )
                else:
                    ax.set_ylabel(zlabels[row])

            if powers is not None and row == num_planes_to_plot - 1:
                ax.set_xlabel('%d mW' % powers[col], rotation=70)
                
            im = ax.imshow(mean_map[col,:,:,depth_idxs[row]],
                           origin='lower', vmin=min_val, vmax=max_val, cmap=cmap)

            cbar = grid[0].cax.colorbar(im)
            
        if cbar_labels is not None:
            cbar.set_label(cbar_labels[mean_idx], rotation=90, loc='top')

        
def denoise_pscs_in_batches(psc, denoiser, batch_size=4096):

    num_pscs = psc.shape[0]
    num_batches = np.ceil(num_pscs / batch_size)
    den_psc_batched = [denoiser(batch, verbose=False)
                       for batch in np.array_split(psc, num_batches, axis=0)]
    return np.concatenate(den_psc_batched)

def estimate_spike_waveforms(lam, den_psc):
    lr = Ridge(fit_intercept=False, alpha=1e-3)
    lr.fit(lam.T, den_psc)
    return lr.coef_.T

def make_grid_latencies(grid_waveforms):
    grid_dims = grid_waveforms.shape[:-1]
    wv_flat = np.reshape(grid_waveforms, (-1, 900))
    max_vals = np.max(wv_flat, -1)

    # create mask which is 1 whenever a PSC is above the threshold
    mask = wv_flat >= 0.1 * max_vals[...,None]

    # use argmax to get the first nonzero entry per row
    first_nonzero_idxs = (mask).argmax(axis=-1)

    # convert index to time in milliseconds
    sample_khz = 20
    samples_per_sec = sample_khz * 1e3
    secs_per_sample = 1 / samples_per_sec
    msecs_per_sample = secs_per_sample * 1e3
    latencies_flat = msecs_per_sample * first_nonzero_idxs

    return latencies_flat.reshape(*grid_dims)

def plot_spike_inference_with_waveforms(den_psc, stim, I, model, waveforms, latencies,
                         spike_thresh=0.01, save=None, ymax=None, n_plots=15, num_trials=30, 
                         weights=None, col_width=10.5, row_height=0.6, order=None,
                         title=None, raw_psc=None, fontsize=14):
    N = stim.shape[0]
    K = den_psc.shape[0]
    trial_len = 900
    normalisation_factor = np.max(np.abs(den_psc))
    trace_linewidth = 0.65
    ymax = 1.05
    ymin = -0.05 * ymax
    
    mu = model.state['mu']
    mu /= np.max(mu)
    lam = model.state['lam']
    z = model.state['z']

    fig = plt.figure(figsize=(col_width, row_height * n_plots * 1.5), dpi=300, facecolor='white')
    
    # one columns for inferred waveforms
    powers = np.unique(I)
    width_ratios = np.zeros(len(powers) + 1)
    width_ratios[0:len(powers)] = 1
    width_ratios[-1] = n_plots
    gs = fig.add_gridspec(ncols=len(powers) + 1, nrows=n_plots, hspace=0.5, wspace=0.05, width_ratios=width_ratios)
        
    for m in range(n_plots):
        n = order[m]
        
        # spike predictions
        ax = fig.add_subplot(gs[m,-1])
        
        if title is not None and m == 0:
            plt.title(title, fontsize=fontsize, y=1.5)
            
        
        trials_per_power = num_trials // len(powers)
        stim_locs = np.array([])
        for pwr in powers:
            stim_locs = np.concatenate([stim_locs, np.where(stim[n] == pwr)[0][:trials_per_power]])
            
        stim_locs = stim_locs.astype(int)
        this_y_psc = den_psc[stim_locs].flatten()/normalisation_factor
        n_repeats = np.min([len(stim_locs), num_trials])
        trial_breaks = np.arange(0, trial_len * n_repeats + 1, trial_len)
        
        if raw_psc is not None:
            this_y_psc_raw = raw_psc[stim_locs].flatten()/normalisation_factor

        plt.xlim([0, trial_len*n_repeats])
        
        # if we have ground truth weights
        if weights is None:
            trace_col = 'k'
        else:
            trace_col = 'k' if weights[n] != 0 else 'gray'
            
        facecol = 'firebrick'
        for tb in range(len(trial_breaks) - 1):
            if tb > 0:
                plt.plot([trial_breaks[tb], trial_breaks[tb]], [ymin, ymax], '--', color=trace_col)
                
            ax.fill_between(np.arange(trial_len * tb, trial_len * (tb + 1)), ymin * np.ones(trial_len), ymax * np.ones(trial_len), facecolor=facecol, 
                                 edgecolor='None', alpha=lam[n, stim_locs][tb] * mu[n], zorder=-5)
                           
            if z[stim_locs][tb] != 0:
                plt.plot(trial_len * (tb + 0.5), 0.75 * ymax, marker='*', markerfacecolor='b', markeredgecolor='None', markersize=6)
                
            # Plot power changes
            if (m == 0) and (I[stim_locs][tb] != I[stim_locs][tb-1]):
                plt.text(trial_breaks[tb], 1.1 * ymax, '%i mW'%I[stim_locs][tb], fontsize=fontsize-2)
                
        plt.plot(this_y_psc, color=trace_col, linewidth=trace_linewidth)
        if raw_psc is not None:
            plt.plot(this_y_psc_raw, color='gray', linewidth=trace_linewidth, alpha=0.5)
        
        for loc in ['top', 'right', 'left', 'bottom']:
            plt.gca().spines[loc].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        plt.ylim([ymin, ymax])
#         plt.ylabel(m+1, fontsize=fontsize-1, rotation=0, labelpad=15, va='center')

        ax.set_rasterization_zorder(-2)
        
    if waveforms is not None:
        ### Inferred PSC waveforms ###
        waveform_colors=['blue','green','purple','red']

        for m in range(n_plots):
            n = order[m]
            

            for power_idx, power in enumerate(powers):
                ax = fig.add_subplot(gs[m,power_idx])
                
                plt.plot(waveforms[power_idx, n, :]/normalisation_factor,
                         color=waveform_colors[power_idx], linewidth=trace_linewidth)
                
                # draw vertical line at inferred latency, first convert to index
                sample_khz = 20
                samples_per_sec = sample_khz * 1e3
                secs_per_sample = 1 / samples_per_sec
                msecs_per_sample = secs_per_sample * 1e3
                plt.axvline(x=(latencies[power_idx, n] / msecs_per_sample),
                            color=waveform_colors[power_idx],
                            linewidth=trace_linewidth, linestyle='-.')

                for loc in ['top', 'right', 'left', 'bottom']:
                    plt.gca().spines[loc].set_visible(False)
                plt.xticks([])
                plt.yticks([])
                plt.ylim([ymin, ymax])

    return fig

