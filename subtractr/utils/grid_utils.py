import numpy as np
import matplotlib as mpl

from mpl_toolkits.axes_grid1 import ImageGrid
from pathlib import Path


from sklearn.linear_model import Ridge, LassoCV
import matplotlib.pyplot as plt
import time
import h5py
import matplotlib.patches as patches


def sort_results(results):
    """
    Sort results dictionary so that items are ordered by their (xyz) position in space.
    This is critical for comparing inferred maps between single and multispot data. 
    We assume that the results dictionary is the results of running
    subtract_utils.run_subtraction_pipeline
    """
    results = dict(results)
    model_state = results['model_state'].item()
    targets = results['targets']
    N = targets.shape[0]

    print(targets.shape)

    # ensure that targets are in the right order so that things work when reshaped
    idxs = np.lexsort((targets[:, -1], targets[:, -2], targets[:, -3]))

    # reorder everything in results
    for (key, value) in model_state.items():
        # first condition ensures we don't index into zero-length tuple
        if value.shape and value.shape[0] == N:
            model_state[key] = value[idxs]
    results['model_state'] = model_state
    results['targets'] = targets[idxs]

    return results


def sequential_map(vals):
    uniques = np.unique(vals)
    return {x: i for i, x in enumerate(uniques)}


def sequential_map_nd(vals, axis):
    uniques = np.unique(vals, axis=axis)
    return {tuple(x): i for i, x in enumerate(uniques)}


def make_stim_matrix(I, L):
    # expect L to have shape trials x spots x 3
    num_trials, num_spots, _ = L.shape

    # Now extract unique x, y, z
    L_flat = L.reshape(-1, 3)
    unique_locs = np.unique(L_flat, axis=0)
    loc_map = sequential_map_nd(L_flat, axis=0)
    num_neurons = unique_locs.shape[0]

    stim = np.zeros((num_neurons, num_trials))

    # for each trial, convert the tuple x,y,z to a single liner index (from 0 to num_locs)
    # then set that entry of the stim matrix
    for trial in range(num_trials):
        for spot in range(num_spots):
            neuron_idx = loc_map[tuple(L[trial, spot, :])]
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
    unique_locs, counts = np.unique(np.c_[L, I], axis=0, return_counts=True)
    max_stims = np.max(counts)

    # Now extract unique x, y, z
    xs = np.unique(L[:, 0])
    ys = np.unique(L[:, 1])
    zs = np.unique(L[:, 2])

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
        powerloc_idx = (p_map[I[trial]],
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
        loc_idx = (x_map[L[trial, 0]], y_map[L[trial, 1]], z_map[L[trial, 2]])
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


def make_psc_tensor_multispot(pscs, powers, targets, stim_mat):
    '''
    Stack all observations into a PSC tensor of shape powers x height x width x trials x time.

    If not all pixels were hit the same number of times, pad the array with nan.

    arguments:
        L: array of size (num_stims x num_spots x 3).
        I: array of size (num_stims)
        stim: array of size (num_pixels x num_stims)
    '''
    num_trials = powers.shape[0]
    unique_powers = np.unique(powers)
    num_powers = len(unique_powers)
    timesteps = pscs.shape[-1]

    # First, we need to compute the maximum number of times a pixel was stimmed.
    # If a pixel was stimmed 10x at 30mW and 10x at 50mW, we want to count these separately,
    # so we stack powers and locations to find stims at each unique power.
    max_stims = get_max_hits(powers, stim_mat)

    # Now extract unique x, y, z
    xs = np.unique(targets[:, 0])
    ys = np.unique(targets[:, 1])
    zs = np.unique(targets[:, 2])

    # create maps to map from location in real coordinates to location in index coordinates
    x_map = sequential_map(xs)
    y_map = sequential_map(ys)
    z_map = sequential_map(zs)
    p_map = sequential_map(unique_powers)

    # Create array and fill with nan
    psc_tensor = np.zeros((num_powers, len(xs), len(
        ys), len(zs), max_stims, timesteps)) + np.nan
    stim_inds = np.zeros((num_powers, len(xs), len(ys), len(zs)), dtype=int)

    for trial in range(num_trials):
        neurons_this_trial = np.where(stim_mat[:, trial])[0]
        for neuron_idx in neurons_this_trial:
            x_idx = x_map[targets[neuron_idx, 0]]
            y_idx = y_map[targets[neuron_idx, 1]]
            z_idx = z_map[targets[neuron_idx, 2]]
            p_idx = p_map[powers[trial]]
            powerloc_idx = (p_idx, x_idx, y_idx, z_idx)

            stim_idx = stim_inds[powerloc_idx]
            combined_idx = (*powerloc_idx, stim_idx)
            psc_tensor[combined_idx] = pscs[trial]

            # increment number of stims for a given location at a given power
            stim_inds[powerloc_idx] += 1

    return psc_tensor


def stack_observations_in_grid(y, I, L):
    num_trials = L.shape[0]
    power_levels = np.unique(I)

    xs = np.unique(L[:, 0])
    ys = np.unique(L[:, 1])
    zs = np.unique(L[:, 2])

    # create arrays to store receptive fields
    x_map = sequential_map(xs)
    y_map = sequential_map(ys)
    z_map = sequential_map(zs)
    p_map = sequential_map(power_levels)

    obs = np.empty((len(power_levels), len(
        xs), len(ys), len(zs)), dtype=object)
    # initialize everything to the empty list.
    # unfortunately there's no easy way to do it.
    for pidx in range(len(power_levels)):
        for xidx in range(len(xs)):
            for yidx in range(len(ys)):
                for zidx in range(len(zs)):
                    obs[pidx, xidx, yidx, zidx] = []

    num_stims = np.zeros_like(obs, dtype=float)

    for trial in range(num_trials):
        idx = (p_map[I[trial]],
               x_map[L[trial, 0]],
               y_map[L[trial, 1]],
               z_map[L[trial, 2]],
               )

        obs[idx].append(y[trial])
        num_stims[idx] += 1

    return obs, num_stims


def make_suff_stats(y, I, L):
    obs, num_stims = stack_observations_in_grid(y, I, L)
    obs_mean = np.vectorize(lambda x: 0 if x == [] else np.mean(x))(obs)
    obs_var = np.vectorize(lambda x: 0 if x == [] else np.var(x))(obs)

    return obs_mean, num_stims, obs_var


def denoise_pscs_in_batches(psc, denoiser, batch_size=4096):

    num_pscs = psc.shape[0]
    num_batches = np.ceil(num_pscs / batch_size)
    den_psc_batched = [denoiser(batch, verbose=False)
                       for batch in np.array_split(psc, num_batches, axis=0)]
    return np.concatenate(den_psc_batched)


def make_grid_waveforms(model_state, psc, powers, grid_dims):

    # here we assume all pixels in the grid were hit with
    # the same powers
    unique_powers = np.unique(powers)  # powers for first plane

    psc_length = psc[0].shape[-1]
    # e.g 2, 26, 26, 5, 900
    grid_waveforms = np.zeros((len(unique_powers), *grid_dims, psc_length))

    for power_idx, power in enumerate(unique_powers):

        # extract lambda and pscs for current power
        these_stims = powers == power

        lam_curr = model_state['lam'][:, these_stims]
        den_psc_curr = psc[these_stims, :]

        curr_waveforms = estimate_spike_waveforms(lam_curr, den_psc_curr)
        curr_waveforms = curr_waveforms.reshape(
            grid_dims[0], grid_dims[1], grid_dims[2], psc_length)

        grid_waveforms[power_idx, :, :, :, :] = curr_waveforms

    return grid_waveforms


def estimate_spike_waveforms(lam, den_psc):
    lr = Ridge(fit_intercept=False, alpha=1e-3)
    lr.fit(lam.T, den_psc)
    return lr.coef_.T


def make_grid_latencies(grid_waveforms,
                        onset_frac=0.1,
                        stim_start=100,
                        srate_khz=20):

    grid_dims = grid_waveforms.shape[:-1]
    wv_flat = np.reshape(grid_waveforms, (-1, 900))
    max_vals = np.max(wv_flat, -1)
    disconnected_idx = (np.sum(wv_flat, axis=-1) == 0)

    # create mask which is 1 whenever a PSC is above the threshold
    mask = wv_flat >= onset_frac * max_vals[..., None]

    # use argmax to get the first nonzero entry per row
    first_nonzero_idxs = (mask).argmax(axis=-1)

    # convert index to time in milliseconds
    secs_per_sample = 1 / (srate_khz * 1e3)
    msecs_per_sample = secs_per_sample * 1e3
    latencies_flat = msecs_per_sample * first_nonzero_idxs

    # set disconnected idxs to nan for plotting convenience
    latencies_flat[disconnected_idx] = np.nan

    # calc amount of time before stim, subtract it off
    # to account for pre-stim context
    pre_stim_time_ms = msecs_per_sample * stim_start
    latencies_flat -= pre_stim_time_ms

    return latencies_flat.reshape(*grid_dims)


def reshape_lasso_response(resp, targets, grid_dims):
    
    num_powers, num_points = resp.shape
    if num_powers * num_points == np.prod(grid_dims):
        return np.reshape(resp, (num_powers, *grid_dims))

    # If some points are missing from the grid
    # e.g if the center points have been removed,
    # we will have to manually fill in the response.
    xs = np.unique(targets[:,0])
    ys = np.unique(targets[:,1])
    zs = np.unique(targets[:,2])
    resp_out = np.zeros((num_powers, *grid_dims))
    for pidx in range(num_powers):
        resp_this_power = resp[pidx]
        target_map = dict([(tuple(k),v) for (k,v) in zip(targets, resp_this_power)])

        for i,x in enumerate(xs):
            for j,y in enumerate(ys):
                for k,z in enumerate(zs):
                    resp_out[pidx, i, j, k] = target_map.get((x,y,z), 0.0)
    return resp_out

    

def circuitmap_lasso_cv(stim_mat, pscs, K=5):
    """
    For each power, return a list of inferred weights via L1 regularized regression.
    The L1 penalty is determined by K-fold cross validation.

    returns:
        responses: (num_powers x num_cells) inferred weight at each power
    """
    powers = np.unique(stim_mat.ravel())[1:]  # exclude zero
    num_powers = len(powers)
    N = stim_mat.shape[0]
    responses = np.zeros((num_powers, N))
    models = []
    for pidx, power in enumerate(powers):

        curr_trials = np.max(stim_mat, axis=0) == power
        curr_stim = stim_mat[:, curr_trials]
        curr_pscs = pscs[curr_trials, :]
        stim_binarized = curr_stim > 0

        # fit lasso with cross val
        print('Starting lasso CV')
        start_time = time.time()
        y = np.trapz(curr_pscs, axis=-1)
        model = LassoCV(cv=K, n_jobs=-1, positive=True,
                        fit_intercept=False, n_alphas=10).fit(stim_binarized.T, y)
        end_time = time.time() - start_time
        print('CV at single power took %f secs' % end_time)
        responses[pidx] = model.coef_
        models.append(model)

    return responses, models

# separate data by pulses


def separate_by_pulses(stim_mat, pscs, npulses):
    stim_mat_list = [stim_mat[:, i::npulses] for i in range(npulses)]
    pscs_list = [pscs[i::npulses, :] for i in range(npulses)]
    powers_list = [np.max(x, axis=0) for x in stim_mat_list]
    return stim_mat_list, pscs_list, powers_list


def load_h5_data(dataset_path, pulse_idx=-1):
    with h5py.File(dataset_path) as f:
        pscs = np.array(f['pscs']).T
        stim_mat = np.array(f['stimulus_matrix']).T
        targets = np.array(f['targets']).T
        powers = np.max(stim_mat, axis=0)

        if 'num_pulses_per_holo' in f:
            npulses = np.array(f['num_pulses_per_holo'], dtype=int).item()
        else:
            npulses = 1

        # get rid of any trials where we didn't actually stim
        good_idxs = (powers > 0)
        pscs = pscs[good_idxs, :]
        stim_mat = stim_mat[:, good_idxs]
        powers = powers[good_idxs]

        # return values corresponding to the pulse we want,
        # in case of multipulse experimental design
        stim_mat_list, pscs_list, powers_list = separate_by_pulses(
            stim_mat, pscs, npulses=npulses)
        stim_mat = stim_mat_list[pulse_idx]
        pscs = pscs_list[pulse_idx]
        powers = powers_list[pulse_idx]

        return pscs, stim_mat, powers, targets
