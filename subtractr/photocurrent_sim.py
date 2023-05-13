import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.scipy as jsp
import subtractr.psc_sim as psc_sim
import scipy.signal as sg

from jax import vmap, jit
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from functools import partial
from typing import Union


def monotone_decay_filter(arr, monotone_start=500):
    ''' Enforce monotone decay beyond kwarg monotone_start. Performed in-place by default.
    '''

    def f(carry, t):
        carry = jnp.minimum(carry, arr[:,t])
        return carry, carry
        
    return (jax.lax.scan(f, arr[:,monotone_start], np.arange(monotone_start, arr.shape[1]))[1]).T
    

def _photocurrent_shape(
    O_inf, R_inf, tau_o, tau_r, g,  # shape params
    t_on, t_off,  # timing params
    linear_onset=False, # boolean, whether or not we use a linear onset
    t=None,
    time_zero_idx=None,
    O_0=0.0, R_0=1.0,
    window_len=900,
    msecs_per_sample=0.05,
    conv_window_len=20,
):

    # In order to correctly handle stim times which start at t < 0,
    # we need to work with a larger window and then crop later.
    # We also pad the window by conv_window_len to avoid zero
    # padding issues during the convolution step.
    # left_bound = jnp.minimum(0, t_on / msecs_per_sample)
    # right_bound = jnp.abs(left_bound) + window_len + conv_window_len
    # t = jnp.arange(left_bound, right_bound) * msecs_per_sample

    # # get the index where t=0 occurs. This is the beginning of the
    # # window we'll return to the user.
    # time_zero_idx = int(-jnp.minimum(t_on / msecs_per_sample, 0))

    mask_stim_on = jnp.where((t >= t_on) & (t <= t_off), 1, 0)
    mask_stim_off = jnp.where((t > t_off), 1, 0)

    # get index where stim is off
    index_t_off = time_zero_idx + jnp.array(t_off // msecs_per_sample, dtype=int)    

    O_on = mask_stim_on * (O_inf - (O_inf - O_0) *
                           jnp.exp(- (t - t_on)/(tau_o)))
    O_off = mask_stim_off * O_on[index_t_off] * jnp.exp(-(t - t_off)/tau_o)

    R_on = mask_stim_on * (R_inf - (R_inf - R_0) * jnp.exp(-(t - t_on)/tau_r))
    R_off = mask_stim_off * \
        (1 - (1 - R_on[index_t_off]) * jnp.exp(-(t - t_off)/tau_r))

    # form photocurrent from each part
    i_photo = g * (O_on + O_off) * (R_on + R_off)


    # if linear_onset=True, use a different version of i_photo with the rising
    # period replaced.
    i_photo_linear = jnp.copy(i_photo)
    stim_off_val = i_photo_linear[index_t_off]
    # zero out the current during the stim
    i_photo_linear = i_photo_linear - (i_photo * mask_stim_on)
    # add linear onset back in
    i_photo_linear = i_photo_linear + ((t - t_on) / (t - t_on)[index_t_off] * stim_off_val) * mask_stim_on

    # conditionally replace i_photo
    i_photo = jax.lax.cond(
        linear_onset,
        lambda _:i_photo_linear,
        lambda _:i_photo,
        None,
    )
        
    # convolve with gaussian to smooth
    x = jnp.linspace(-3, 3, conv_window_len)
    window = jsp.stats.norm.pdf(x, scale=25)
    window = window.at[0:conv_window_len//2].set(0)
    window = window / window.sum()
    i_photo = jsp.signal.convolve(i_photo, window, mode='same')
    i_photo /= (jnp.max(i_photo) + 1e-3)

    return (i_photo[time_zero_idx:time_zero_idx + window_len],
            O_on[time_zero_idx:time_zero_idx + window_len],
            O_off[time_zero_idx:time_zero_idx + window_len],
            R_on[time_zero_idx:time_zero_idx + window_len],
            R_off[time_zero_idx:time_zero_idx + window_len])

photocurrent_shape = jax.jit(_photocurrent_shape, static_argnames=('time_zero_idx', 'window_len', 'msecs_per_sample', 'conv_window_len'))

def _sample_photocurrent_params(
    key,
    t_on_min=5.0,
    t_on_max=7.0,
    t_off_min=10.0,
    t_off_max=11.0,
    O_inf_min=0.3,
    O_inf_max=1.0,
    R_inf_min=0.3,
    R_inf_max=1.0,
    tau_o_min=5,
    tau_o_max=14,
    tau_r_min=25,
    tau_r_max=30,):
    keys = jrand.split(key, num=6)

    t_on  = jrand.uniform(keys[0], minval=t_on_min, maxval=t_on_max)
    t_off  = jrand.uniform(keys[1], minval=t_off_min, maxval=t_off_max)
    O_inf = jrand.uniform(keys[2], minval=O_inf_min, maxval=O_inf_max)
    R_inf = jrand.uniform(keys[3], minval=R_inf_min, maxval=R_inf_max)
    tau_o = jrand.uniform(keys[4], minval=tau_o_min, maxval=tau_o_max)
    tau_r = jrand.uniform(keys[5], minval=tau_r_min, maxval=tau_r_max)
    g = 1.0

    return O_inf, R_inf, tau_o, tau_r, g, t_on, t_off

def _sample_photocurrent_params_hierarchical(
    key,
    num_traces=32,
    t_on_min=5.0,
    t_on_max=5.2,
    t_off_min=10.0,
    t_off_max=10.2,
    O_inf_min=0.3,
    O_inf_max=1.0,
    R_inf_min=0.3,
    R_inf_max=1.0,
    tau_o_min=5,
    tau_o_max=14,
    tau_r_min=25,
    tau_r_max=30,):

    keys = iter(jrand.split(key, num=6))

    # Sample shape parameters shared across all experiments
    O_inf = jrand.uniform(next(keys),
        minval=O_inf_min, maxval=O_inf_max) * jnp.ones(num_traces,)
    R_inf = jrand.uniform(next(keys),
        minval=R_inf_min, maxval=R_inf_max) * jnp.ones(num_traces,)
    tau_o = jrand.uniform(next(keys),
        minval=tau_o_min, maxval=tau_o_max) * jnp.ones(num_traces,)
    tau_r = jrand.uniform(next(keys),
        minval=tau_r_min, maxval=tau_r_max) * jnp.ones(num_traces,)
    g = 1.0 * jnp.ones(num_traces)

    # sample timing parameters unique to each trace in the experiment
    t_on  = jrand.uniform(next(keys), minval=t_on_min, maxval=t_on_max, shape=(num_traces,))
    t_off  = jrand.uniform(next(keys), minval=t_off_min, maxval=t_off_max, shape=(num_traces,))

    return tuple(jnp.row_stack((O_inf, R_inf, tau_o, tau_r, g, t_on, t_off,)))

def _sample_scales(key, min_pc_fraction, max_pc_fraction,
                   num_traces, min_pc_scale, max_pc_scale):

    keys = iter(jrand.split(key, num=5))

    # sample scale values for photocurrents.
    # Randomly set some traces to have no photocurrent
    # according to pc_fraction.
    pc_fraction = jrand.uniform(next(keys), minval=min_pc_fraction,
                                maxval=max_pc_fraction)
    pc_mask = jnp.where(
        jrand.uniform(next(keys), shape=(num_traces,)) <= pc_fraction,
        1.0,
        0.0)

    # draw scales centered on a random value
    # pc_scale_center = jrand.uniform(next(keys), minval=min_pc_scale, maxval=max_pc_scale)
    # pc_scales = jrand.normal(next(keys), shape=(num_traces,)) + pc_scale_center
    # pc_scales = jnp.clip(pc_scales, a_min=min_pc_scale, a_max=max_pc_scale)
    pc_scales = jrand.uniform(next(keys), shape=(num_traces,),
        minval=min_pc_scale, maxval=max_pc_scale)

    pc_scales *= pc_mask
    return pc_scales


def _sample_gp(key, pcs, gp_lengthscale=25, gp_scale=0.01, ):
    n_samples, trial_dur = pcs.shape
    # creates a distance matrix between indices,
    # much faster than a loop
    D = jnp.broadcast_to(jnp.arange(trial_dur), (trial_dur, trial_dur))
    D -= jnp.arange(trial_dur)[:, None]
    D = jnp.array(D, dtype=jnp.float64)
    K = jnp.exp(-D**2/(2 * gp_lengthscale**2)) + 1e-4 * jnp.eye(trial_dur)
    mean = jnp.zeros(trial_dur, dtype=jnp.float64)
    return gp_scale * jrand.multivariate_normal(key, mean=mean, cov=K, shape=(n_samples,))
sample_gp = jax.jit(_sample_gp)


@jax.jit
def _sample_experiment_noise_and_scales(
    key,
    cur_pc_template,
    prev_pc_template,
    next_pc_template,
    psc_background,
    min_pc_scale,
    max_pc_scale,
    min_pc_fraction,
    max_pc_fraction,
    prev_pc_fraction,
    gp_lengthscale,
    gp_scale,
    iid_noise_scale,
):
    num_traces, trial_dur = psc_background.shape

    keys = jrand.split(key, num=4)
    prev_pcs = _sample_scales(keys[0],
                              prev_pc_fraction, prev_pc_fraction,
                              num_traces, min_pc_scale, max_pc_scale)[:, None] * prev_pc_template

    cur_pcs = _sample_scales(keys[1],
                             min_pc_fraction, max_pc_fraction,
                             num_traces, min_pc_scale, max_pc_scale)[:, None] * cur_pc_template

    next_pcs = _sample_scales(keys[2],
                              min_pc_fraction, max_pc_fraction,
                              num_traces, min_pc_scale, max_pc_scale)[:, None] * next_pc_template

    # TODO: add GP and IID noise
    gp_noise = sample_gp(
        keys[3],
        cur_pcs,
        gp_lengthscale=gp_lengthscale,
        gp_scale=gp_scale,
    )

    iid_noise = iid_noise_scale * \
        jrand.normal(keys[3], shape=(num_traces, trial_dur))
    targets = cur_pcs
    observations = prev_pcs + cur_pcs + next_pcs + iid_noise + gp_noise + psc_background

    return observations, targets

def _default_pc_shape_params():
    return dict(
        # shape params
        O_inf_min=0.3,
        O_inf_max=1.0,
        R_inf_min=0.3,
        R_inf_max=1.0,
        tau_o_min=3,
        tau_o_max=8,
        tau_r_min=25,
        tau_r_max=30,
    )


def _exp_func(t, a, b, c):
    ''' Exponential function
    '''
    return a * np.exp(b * t) + c


def _fit_exponential_tail(trace, t, a0, b0, c0):
    '''
    Fit exponentials to the provided traces.
    params:
        traces: N x T array
    returns:
        a, b, c: length N arrays of parameters, such that the exponential
                 a[i] * exp(b[i] * t) + c[i] ~= traces[i] for i = 1,...,N
    '''
    popt, pcov = curve_fit(
        _exp_func,
        t, trace,
        p0=(a0, b0, c0)
    )
    return popt


def _extend_traces(
    traces,
    msecs_per_sample,
    num_samples_to_add,
    fit_start_idx=400,
    replace_start_idx=600,
    a0=0.1,
    b0=-1.0/20.0,
    c0=0.5
    ):
    N, window_len = traces.shape
    t_fit = np.arange(fit_start_idx, window_len) * msecs_per_sample
    params = [_fit_exponential_tail(trace, t_fit, a0, b0, c0)
        for trace in traces[:, fit_start_idx:]]
    
    # create decaying exponentials of length num_samples_to_add
    t_new = np.arange(replace_start_idx, window_len + num_samples_to_add) \
         * msecs_per_sample
    extensions = np.array([_exp_func(t_new, *popt) for popt in params])

    # concatenate traces with estimated tails
    out = np.zeros((N, window_len + 2 * num_samples_to_add))
    out[:, num_samples_to_add:num_samples_to_add + replace_start_idx] = traces[:, 0:replace_start_idx]
    out[:, num_samples_to_add + replace_start_idx:] = extensions

    # extend beginning of trace by appending constant
    out[:, 0:num_samples_to_add] = traces[:,0:1]

    return out


def sample_from_templates(
    templates,
    key,
    size=100,
    jitter_ms=0.5,
    window_len=900,
    smoothing_sigma=5,
    max_scaling_frac=0.5,
    msecs_per_sample=0.05,
    stim_start=100,
    exponential_fit_start_idx=450,
    add_target_gp=False,
    target_gp_lengthscale=50,
    target_gp_scale=0.01,
    ):
    '''
    sample traces from templates with augmentation by jitter and scaling
    '''

    templates[:, 0:stim_start] = 0.0

    # extend templates so that we can sample using jitter
    num_samples_to_add =  int(np.round(jitter_ms / msecs_per_sample))
    extended_traces = _extend_traces(
        templates,
        msecs_per_sample,
        num_samples_to_add,
        exponential_fit_start_idx,
    )
    extended_traces_smoothed = gaussian_filter1d(
        extended_traces, sigma=smoothing_sigma)
    
    out = np.zeros((size, templates.shape[-1]))
    for i in range(size):
        this_template_idx = np.random.randint(templates.shape[0])
        this_template = np.copy(extended_traces_smoothed[this_template_idx])
        this_scale = 1.0 + np.random.uniform(low=-max_scaling_frac, high=max_scaling_frac)
        this_template *= this_scale

        # sample jitter in number of samples to shift
        this_jitter_samples = np.random.randint(low=0, high=num_samples_to_add)
        start_idx = num_samples_to_add + this_jitter_samples
        out[i] = this_template[start_idx:start_idx + window_len]

    if add_target_gp:
        stim_start_idx = int(stim_start // msecs_per_sample)
        key = jrand.fold_in(key, 0)
        target_gp = np.array(_sample_gp(
            key, 
            out,
            gp_lengthscale=target_gp_lengthscale,
            gp_scale=target_gp_scale,
        ))
        target_gp = np.maximum(0, target_gp)
        out = np.array(out)
        out[:, stim_start_idx+10:] += target_gp[:, stim_start_idx+10:]
        out = monotone_decay_filter(
            out,
        )

    return out

@partial(jit, static_argnames=('num_traces', 'msecs_per_sample', 'stim_start', 'stim_end', 'isi_ms', 'window_len_ms', 'add_target_gp'))
def sample_jittered_photocurrent_shapes(
        key : jax.random.PRNGKey, 
        num_traces: int,
        onset_jitter_ms: float = 2.0,
        onset_latency_ms: float = 0.2,
        msecs_per_sample: float = 0.05,
        stim_start: float = 5.0,
        stim_end: float = 10.0,
        isi_ms: float = 33,
        window_len_ms: float = 45,
        pc_shape_params: Union[dict, None] = None,
        linear_onset_frac: float = 0.5,
        add_target_gp: float = True,
        target_gp_lengthscale: float = 50,
        target_gp_scale: float = 0.01,
        monotone_filter_start: float = 300,
    ):
    keys = iter(jrand.split(key, num=10))
    if pc_shape_params is None:
        pc_shape_params = _default_pc_shape_params()

    # generate all photocurrent templates.
    # We create a separate function to sample each of previous, current, and
    # next PSC shapes.
    prev_pc_params = _sample_photocurrent_params_hierarchical(
            next(keys),
            num_traces=num_traces,
            **pc_shape_params,
            # t_on_min=-28.0 + onset_latency_ms, t_on_max=-28.0 + onset_latency_ms + onset_jitter_ms,
            # t_off_min=-23.0 + onset_latency_ms, t_off_max=-23.0 + onset_latency_ms + onset_jitter_ms,
            t_on_min = stim_start - isi_ms + onset_latency_ms, t_on_max = stim_start - isi_ms + onset_latency_ms + onset_jitter_ms,
            t_off_min = stim_end - isi_ms + onset_latency_ms, t_off_max = stim_end - isi_ms + onset_latency_ms + onset_jitter_ms,

        )

    curr_pc_params = _sample_photocurrent_params_hierarchical(
            next(keys),
            num_traces=num_traces,
            **pc_shape_params,
            # t_on_min=5.0 + onset_latency_ms, t_on_max=5.0 + onset_latency_ms + onset_jitter_ms,
            # t_off_min=10.0 + onset_latency_ms, t_off_max=10.0 + onset_latency_ms + onset_jitter_ms,
            t_on_min = stim_start + onset_latency_ms, t_on_max = stim_start + onset_latency_ms + onset_jitter_ms,
            t_off_min = stim_end + onset_latency_ms, t_off_max = stim_end + onset_latency_ms + onset_jitter_ms,
        )

    next_pc_params = _sample_photocurrent_params_hierarchical(
            next(keys),
            num_traces=num_traces,
            **pc_shape_params,
            # t_on_min=38.0 + onset_latency_ms, t_on_max=38.0 + onset_latency_ms + onset_jitter_ms,
            # t_off_min=43.0 + onset_latency_ms, t_off_max=43.0 + onset_latency_ms + onset_jitter_ms,
            t_on_min = stim_start + isi_ms + onset_latency_ms, t_on_max = stim_start + isi_ms + onset_latency_ms + onset_jitter_ms,
            t_off_min = stim_end + isi_ms + onset_latency_ms, t_off_max = stim_end + isi_ms + onset_latency_ms + onset_jitter_ms,
        )
    
    # form boolean for each trace deciding whether to use linear onset
    linear_onset = jrand.uniform(key) > linear_onset_frac

    # Note that we simulate using a longer window than we'll eventually use. This allows us to compute the previous and next trial shapes
    # time = jnp.arange(tstart / msecs_per_sample, tend / msecs_per_sample) * msecs_per_sample
    tstart = stim_start - isi_ms - 5.0
    tend = max(stim_end + isi_ms, window_len_ms) + 5.0 
    time = jnp.arange(tstart, tend, msecs_per_sample)
    time_zero_idx = int(-tstart / msecs_per_sample)
    batched_photocurrent_shape = vmap(
        partial(
            photocurrent_shape,
            linear_onset=linear_onset,
            t=time,
            time_zero_idx=time_zero_idx,
            window_len=int(window_len_ms / msecs_per_sample),
        ),
        # in_axes=(0, 0, 0, 0, 0, 0, 0, 0)
    )

    # import pdb; pdb.set_trace()
    prev_pc_shapes = batched_photocurrent_shape(*prev_pc_params)[0]
    curr_pc_shapes = batched_photocurrent_shape(*curr_pc_params)[0]
    next_pc_shapes = batched_photocurrent_shape(*next_pc_params)[0]

    # Add variability to target waveforms to account for mis-specification of 
    # photocurrent model.
    if add_target_gp:
        stim_start_idx = int(stim_start // msecs_per_sample)
        key = jrand.fold_in(key, 0)
        target_gp = _sample_gp(
            next(keys), 
            curr_pc_shapes[0:1,:],
            gp_lengthscale=target_gp_lengthscale,
            gp_scale=target_gp_scale,
        )
        target_gp = jnp.maximum(target_gp, 0.0)
        curr_pc_shapes = curr_pc_shapes.at[:, stim_start_idx+10:].add(target_gp[:, stim_start_idx+10:])
        curr_pc_shapes = curr_pc_shapes.at[:, monotone_filter_start:].set(
            monotone_decay_filter(curr_pc_shapes, monotone_start=monotone_filter_start))

    return prev_pc_shapes, curr_pc_shapes, next_pc_shapes

@partial(jit, static_argnames=(
    'add_target_gp', 'msecs_per_sample', 'num_traces',
    'stim_start', 'stim_end', 'isi_ms', 'window_len_ms', 'normalize_type',
    'inhibitory_pscs'))
def sample_photocurrent_experiment(
    key, num_traces=32, 
    onset_jitter_ms=1.0,
    onset_latency_ms=0.2,
    pc_shape_params=None,
    psc_shape_params=None,
    min_pc_scale = 0.05,
    max_pc_scale = 10.0,
    min_pc_fraction = 0.1,
    max_pc_fraction = 0.95,
    min_prev_pc_fraction = 0.1,
    max_prev_pc_fraction = 0.3,
    add_target_gp=True,
    target_gp_lengthscale=25.0,
	target_gp_scale=0.01,
    linear_onset_frac=0.5,
    msecs_per_sample=0.05,
    stim_start=5.0,
    stim_end=10.0,
    isi_ms=33.0,
    window_len_ms=45.0,
    gp_lengthscale_min=20, 
    gp_lengthscale_max=60,
    gp_scale_min=0.01,
    gp_scale_max=0.05,
    iid_noise_std_min=0.001,
    iid_noise_std_max=0.02,
    normalize_type='max',
    inhibitory_pscs=False,
    ):
    keys = iter(jrand.split(key, num=12))

    if pc_shape_params is None:
        pc_shape_params = dict(
           O_inf_min=0.3,
            O_inf_max=1.0,
            R_inf_min=0.1,
            R_inf_max=1.0,
            tau_o_min=3,
            tau_o_max=3,
            tau_r_min=30,
            tau_r_max=30, 
        )

    if psc_shape_params is None:
        psc_shape_params = dict(
            tau_r_lower = 10,
            tau_r_upper = 40,
            tau_diff_lower = 60,
            tau_diff_upper = 120,
            trial_dur=900,
            delta_lower=160,
            delta_upper=400,
            next_delta_lower=400,
            next_delta_upper=899,
            prev_delta_upper=150,
            amplitude_lower=0.01,
            amplitude_upper=0.5
        )

    # Sample photocurrent waveform and scale randomly
    prev_pc_shapes, curr_pc_shapes, next_pc_shapes = \
			sample_jittered_photocurrent_shapes(
				next(keys),
				num_traces,
				onset_jitter_ms=onset_jitter_ms,
				onset_latency_ms=onset_latency_ms,
                msecs_per_sample=msecs_per_sample,
				pc_shape_params=pc_shape_params,
				add_target_gp=add_target_gp,
				target_gp_lengthscale=target_gp_lengthscale,
				target_gp_scale=target_gp_scale,
				linear_onset_frac=linear_onset_frac,
                stim_start=stim_start,
                stim_end=stim_end,
                isi_ms=isi_ms,
                window_len_ms=window_len_ms,
                )
    max_pc_scale = jrand.uniform(next(keys), minval=min_pc_scale, maxval=max_pc_scale)

    prev_pc_scales = _sample_scales(
        next(keys), min_prev_pc_fraction, max_prev_pc_fraction, num_traces, min_pc_scale, max_pc_scale
    )
    prev_pcs = prev_pc_shapes * prev_pc_scales[:,None]

    curr_pc_scales = _sample_scales(
        next(keys), min_pc_fraction, max_pc_fraction, num_traces, min_pc_scale, max_pc_scale
    )
    curr_pcs = curr_pc_shapes * curr_pc_scales[:,None]

    next_pc_scales = _sample_scales(
        next(keys), min_prev_pc_fraction, max_prev_pc_fraction, num_traces, min_pc_scale, max_pc_scale
    )
    next_pcs = next_pc_shapes * next_pc_scales[:,None]

    # Sample batch of PSC background traces. Fold in keyword args
    # first since vmap doesn't like them.
    _sample_pscs_partial = partial(psc_sim._sample_pscs_single_trace, **psc_shape_params)
    sample_pscs_batch = vmap(_sample_pscs_partial)

    psc_keys = jrand.split(next(keys), num=num_traces)
    pscs, _ = sample_pscs_batch(psc_keys)

    if inhibitory_pscs:
        pscs = -pscs

    # sample noise. Each experiment has it's own noise scales
    gp_scale = jrand.uniform(next(keys), minval=gp_scale_min, maxval=gp_scale_max)
    gp_lengthscale = jrand.uniform(next(keys), minval=gp_lengthscale_min, maxval=gp_lengthscale_max)
    gp_noise = jnp.squeeze(_sample_gp(next(keys), pscs, gp_lengthscale, gp_scale))
    iid_noise_std = jrand.uniform(next(keys), minval=iid_noise_std_min, maxval=iid_noise_std_max)
    iid_noise = jrand.normal(next(keys), shape=pscs.shape) * iid_noise_std

    # combine all ingredients and normalize
    input = pscs + prev_pcs + curr_pcs + next_pcs
    target = curr_pcs

    if normalize_type == 'l2':
        maxv = (jnp.linalg.norm(input) + 1e-5 / num_traces)
    elif normalize_type == 'max':
        maxv = jnp.max(input, axis=-1, keepdims=True) + 1e-5
    elif normalize_type == 'none':
        maxv = jnp.ones((num_traces, 1))
    else:
        raise ValueError('unknown value for normalize_type')

    input /= maxv
    target /= maxv

    # add GP and IID noise after normalizing
    input = input + iid_noise + gp_noise

    return (input, target)


def postprocess_photocurrent_experiment_batch(inputs, lp_cutoff=500, msecs_per_sample=0.05):
    nbatch, ntrace, ntimesteps = inputs.shape
    inputs = inputs.reshape(-1, ntimesteps)
    sos = sg.butter(4, lp_cutoff, btype='low', fs=int(1/msecs_per_sample*1000), output='sos')
    out = sg.sosfiltfilt(sos, inputs, axis=-1)
    out = out.reshape(nbatch, ntrace, ntimesteps) 
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test the sample_jittered_photocurrent_shapes function
    key = jrand.PRNGKey(0)
    prev_pc_shapes, curr_pc_shapes, next_pc_shapes = \
        sample_jittered_photocurrent_shapes(key, num_traces=32, onset_jitter_ms=0.0, onset_latency_ms=0.0 )

    # plot the current shapes with lines for stim onset and offset
    fig = plt.figure(figsize=(3,3), dpi=300)
    fig, ax = plt.subplots(1, 1)

    ax.plot(curr_pc_shapes[0].T)
    ax.axvline(100, color='k', linestyle='--')
    ax.axvline(200, color='k', linestyle='--')
    plt.show()


