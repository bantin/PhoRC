import numpy as np
import phorc

import phorc.simulation.photocurrent_sim as pcsim
import h5py

from scipy.stats import multivariate_normal
from phorc.simulation.photocurrent_sim import sample_jittered_photocurrent_shapes



def add_photocurrents_to_expt(key, expt, pc_shape_params=None,
        frac_pc_cells=0.1, opsin_mean=0.5, opsin_std=0.2,
        stim_dur_ms=5.0, pc_response_var=0.01, pc_window_len_ms=200,
        sampling_freq=20000, stim_freq=30,
        prior_context=100, response_length=900, jitter=False):
    """Add photocurrents to a continuous experiment generated by circuitmap"""

    # calculate the length of the window in samples
    window_len_samples = int(pc_window_len_ms * sampling_freq / 1000)
    assert window_len_samples >= response_length, "Window length must be longer than response length."

    # Draw opsin expression from zero-inflated gamma.
    # The vector opsin_expression determines the _average_ amplitude
    # of an evoked photocurrent when stimulating that cell
    N, K = expt['stim_matrix'].shape
    pc_cells = np.random.choice(N, int(frac_pc_cells*N), replace=False)
    opsin_expression = np.zeros(N)
    opsin_expression[pc_cells] = np.random.gamma(shape=(opsin_mean/opsin_std)**2, scale=opsin_std**2/opsin_mean, size=len(pc_cells))
        
    # Draw actual evoked photocurrent heights centered around their average
    pc_contributions = pc_response_var * (
            np.random.randn(N, K) * (opsin_expression[:, None] > 0)) + opsin_expression[:,None]

    if pc_shape_params is None:
        pc_shape_params = dict(
            O_inf_min=0.3,
                O_inf_max=1.0,
                R_inf_min=0.1,
                R_inf_max=1.0,
                tau_o_min=6,
                tau_o_max=16,
                tau_r_min=5,
                tau_r_max=16, 
            )

    pc_full_params = dict(
        onset_jitter_ms=0.01,
        onset_latency_ms=0.0,
        pc_shape_params=pc_shape_params,
        add_target_gp=False,
        target_gp_lengthscale=20,
        target_gp_scale=0.01,
        linear_onset_frac=1.0,
        msecs_per_sample=0.05,
        stim_start=(prior_context / sampling_freq * 1e3), # calculate stim start based on prior context
        stim_end=(prior_context / sampling_freq * 1e3) + stim_dur_ms,
        isi_ms=1000 / stim_freq,
        window_len_ms=pc_window_len_ms,
    )
   
    # Get a PC shape for every stim
    stim_mat_scaled = expt['stim_matrix'] / np.max(expt['stim_matrix'])

    # Sample photocurrent shapes. optionally add jitter
    if jitter:
        jittered_pc_shapes = np.array(sample_jittered_photocurrent_shapes(key, K, **pc_full_params)[1])
    else:
        jittered_pc_shapes = np.array(sample_jittered_photocurrent_shapes(key, 1, **pc_full_params)[1])
        jittered_pc_shapes = np.broadcast_to(jittered_pc_shapes, (K, jittered_pc_shapes.shape[1]))
    
    pc_scales = np.maximum(np.sum(stim_mat_scaled * pc_contributions, axis=0)[:,None], 0)
    true_photocurrents = pc_scales * jittered_pc_shapes
    
    # Add photocurrents to unrolled trace to capture inter-trial overlap
    obs_flat = unfold_to_flat(expt['obs_responses'],
                    response_length=response_length, prior_context=prior_context,
                        stim_freq=stim_freq, sampling_freq=sampling_freq)
    expt['flat_ground_truth'] = obs_flat.copy()
    
    isi = int(sampling_freq / stim_freq)
    stim_times = np.arange(0, K * isi, isi)

    for i,stim_idx in enumerate(stim_times):
        _end_idx = stim_idx + window_len_samples
        end_idx_flat = np.minimum(_end_idx, len(obs_flat))
        end_idx_wrapped = jittered_pc_shapes.shape[1] - np.maximum(_end_idx - len(obs_flat), 0)
        
        if end_idx_flat <= stim_idx:
            import pdb; pdb.set_trace()
            
        obs_flat[stim_idx:end_idx_flat] += true_photocurrents[i,:end_idx_wrapped]
       
    expt['obs_with_photocurrents'] = fold_overlapping(obs_flat, prior_context, response_length,
                                                         sampling_freq, stim_freq,)
    expt['true_photocurrents'] = true_photocurrents
    expt['opsin_expression'] = opsin_expression
    return expt

def make_spatial_opsin_resp(targets,
    powers,
    stim_mat,
    phi_0=0.1,
    phi_1=50,
    opsin_loc=None):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    unique_powers = np.unique(powers)
    unique_tars = np.unique(targets, axis=0)

    num_xs = len(np.unique(unique_tars[:,0]))
    num_ys = len(np.unique(unique_tars[:,1]))
    num_zs = len(np.unique(unique_tars[:,2]))

    if opsin_loc is None:
        opsin_resp_centroid = np.mean(unique_tars, axis=0)
    else:
        opsin_resp_centroid = opsin_loc
    opsin_resp_cov = np.diag([50, 50, 400])
 
    rv = multivariate_normal(opsin_resp_centroid, opsin_resp_cov)

    opsin_resp = np.zeros(stim_mat.shape[1])
    for trial in range(stim_mat.shape[1]):
        neurons_this_trial = np.where(stim_mat[:, trial])[0]
        for neuron_idx in neurons_this_trial:
            xyz = targets[neuron_idx]
            p = powers[trial]
            opsin_resp[trial] += rv.pdf(xyz) * phi_0 * sigmoid(p - phi_1)

    opsin_resp /= np.max(opsin_resp)
    return opsin_resp


def make_hybrid_spatial_dataset(key,
    pscs,
    targets,
    powers,
    stim_mat,
    opsin_std_dev=0.1,
    pc_shape_params=None,
    onset_latency_ms=0.0,
    onset_jitter_ms=0.2,
    opsin_scale=0.8,
    opsin_loc=None,
    response_cutoff=0.1,
    response_length=900,
    prior_context=100,
    stim_freq=30,
    stim_dur_ms=5,
    sampling_freq=20000,
    pc_window_len_ms=180,):

    if pc_shape_params is None:
        pc_shape_params = dict(
           O_inf_min=0.3,
            O_inf_max=1.0,
            R_inf_min=0.3,
            R_inf_max=1.0,
            tau_o_min=7,
            tau_o_max=7,
            tau_r_min=26,
            tau_r_max=29, 
        )

    pc_full_params = dict(
        onset_jitter_ms=0.01,
        onset_latency_ms=0.0,
        pc_shape_params=pc_shape_params,
        add_target_gp=False,
        target_gp_lengthscale=20,
        target_gp_scale=0.01,
        linear_onset_frac=1.0,
        msecs_per_sample=0.05,
        stim_start=(prior_context / sampling_freq * 1e3), # calculate stim start based on prior context
        stim_end=(prior_context / sampling_freq * 1e3) + stim_dur_ms,
        isi_ms=(1 / stim_freq * 1e3),
        window_len_ms=(response_length / sampling_freq * 1e3),
    )

    # sample a bunch of photocurrent waveforms with the same shape,
    # but with slight jitter
    photocurrent_waveforms = pcsim.sample_jittered_photocurrent_shapes(
            key,
            pscs.shape[0],
            **pc_full_params,
    )
    prev_waveforms, curr_waveforms, next_waveforms = [np.array(x) for x in photocurrent_waveforms]
        
    # create opsin response for previous, current, and next trials
    def get_opsin_scales(targets, powers, stim_mat, opsin_loc, opsin_scale):
        average_opsin_resp = make_spatial_opsin_resp(targets,
            powers, stim_mat, opsin_loc=opsin_loc) * opsin_scale
        opsin_scales = np.random.normal(loc=average_opsin_resp, scale=opsin_std_dev)
        idxs_cutoff = (average_opsin_resp < response_cutoff)
        opsin_scales[idxs_cutoff] = 0.0
        opsin_scales = np.maximum(0, opsin_scales)
        return opsin_scales

    # get opsin scales
    opsin_scales = get_opsin_scales(targets, powers, stim_mat, opsin_loc, opsin_scale)
    
    # combine photocurrents from previous, current, and next trials
    photocurrents = curr_waveforms * opsin_scales[:, None]
    photocurrents[1:] += prev_waveforms[1:] * opsin_scales[:-1, None]
    photocurrents[:-1] += next_waveforms[:-1] * opsin_scales[1:, None]

    pscs_plus_photo = pscs + photocurrents

    return pscs_plus_photo, photocurrents, photocurrent_waveforms


def fold_overlapping(trace, prior_context, response_length, sampling_freq, stim_freq):
    """
    Split trace into overlapping segments of length response_length.
    prior_context determines the amount which is overlapped from the prior trial
    """
    num_samples = trace.shape[0]
    isi = int(sampling_freq / stim_freq)
    start_indices = np.arange(0, num_samples - response_length + prior_context, isi)

    end_indices = start_indices + response_length
    return np.array([trace[x:y] for x,y in zip(start_indices, end_indices)])

def unfold_to_flat(traces, response_length=900, prior_context=100, stim_freq=30, sampling_freq=20000):
    response_length = traces.shape[1]
    
    next_stim_idx = prior_context + int(sampling_freq / stim_freq)
    next_stim_idx = np.minimum(next_stim_idx, response_length-1)
    
        
    # include prior context on first trace, after that we leave it off
    flattened = np.concatenate(traces[:, prior_context:next_stim_idx])
    flattened = np.concatenate((traces[0, 0:prior_context], flattened, traces[-1,next_stim_idx:]))
    
    return flattened
    
    
def subtract_overlapping_trials(orig, est,
            prior_context=100, stim_freq=30,
            sampling_freq=20000, return_flat=False):
    
    
    # compute stim times based on prior context and response length
    num_stims, response_length = orig.shape
    
    obs_flat = unfold_to_flat(orig, response_length=response_length,
                    prior_context=prior_context, stim_freq=stim_freq,
                              sampling_freq=sampling_freq)
    

    isi = int(sampling_freq / stim_freq)
    stim_times = np.arange(0, num_stims * isi, isi)
   
    num_samples = obs_flat.shape[0]
    subtracted_flat = obs_flat.copy()
    for stim_idx, pc_est in zip(stim_times, est):
        _end_idx = stim_idx + response_length
        end_idx_flat = np.minimum(_end_idx, len(obs_flat))
        end_idx_wrapped = pc_est.shape[0] - np.maximum(_end_idx - len(obs_flat), 0)
        subtracted_flat[stim_idx:end_idx_flat] -= pc_est[:end_idx_wrapped]
        
        # # check whether we're at the end of the trace
        # if end_idx > num_samples:
        #     est_cutoff = end_idx - num_samples
        #     end_idx = num_samples
        #     subtracted_flat[stim_idx:end_idx] -= pc_est[:est_cutoff] 
        # else:
        #     subtracted_flat[stim_idx:end_idx] -= pc_est

    if return_flat:
        return subtracted_flat
    return fold_overlapping(subtracted_flat, prior_context, response_length, sampling_freq, stim_freq)
