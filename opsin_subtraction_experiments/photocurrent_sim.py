import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.scipy as jsp
import circuitmap as cm

def _photocurrent_shape(
    O_inf, R_inf, tau_o, tau_r, g,  # shape params
    t_on, t_off,  # timing params
    O_0=0.0, R_0=1.0,
    window_len=900, msecs_per_sample=0.05, conv_window_len=25,
):

    # In order to correctly handle stim times which start at t < 0,
    # we need to work with a larger window and then crop later.
    # We also pad the window by conv_window_len to avoid zero
    # padding issues during the convolution step.
    left_bound = jnp.minimum(0, t_on / msecs_per_sample)
    right_bound = jnp.abs(left_bound) + window_len + conv_window_len
    t = jnp.arange(left_bound, right_bound) * msecs_per_sample

    # get the index where t=0 occurs. This is the beginning of the
    # window we'll return to the user.
    time_zero_idx = int(-jnp.minimum(t_on / msecs_per_sample, 0))

    mask_stim_on = jnp.where((t >= t_on) & (t <= t_off), 1.0, 0.0)
    mask_stim_off = jnp.where((t > t_off), 1.0, 0.0)

    # get index where stim is off
    index_t_off = time_zero_idx + int(jnp.round(t_off / msecs_per_sample))

    O_on = mask_stim_on * (O_inf - (O_inf - O_0) *
                           jnp.exp(- (t - t_on)/(tau_o)))
    O_off = mask_stim_off * O_on[index_t_off] * jnp.exp(-(t - t_off)/tau_o)

    R_on = mask_stim_on * (R_inf - (R_inf - R_0) * jnp.exp(-(t - t_on)/tau_r))
    R_off = mask_stim_off * \
        (1 - (1 - R_on[index_t_off]) * jnp.exp(-(t - t_off)/tau_r))

    # form photocurrent from each part
    i_photo = g * (O_on + O_off) * (R_on + R_off)

    # convolve with gaussian to smooth
    x = jnp.linspace(-3, 3, conv_window_len)
    window = jsp.stats.norm.pdf(x, scale=25)
    i_photo = jsp.signal.convolve(i_photo, window, mode='same')
    i_photo /= (jnp.max(i_photo) + 1e-3)

    return (i_photo[time_zero_idx:time_zero_idx + window_len],
            O_on[time_zero_idx:time_zero_idx + window_len],
            O_off[time_zero_idx:time_zero_idx + window_len],
            R_on[time_zero_idx:time_zero_idx + window_len],
            R_off[time_zero_idx:time_zero_idx + window_len])


def _sample_photocurrent_params(key):
    keys = jax.random.split(key, num=4)

    O_inf, R_inf = jrand.uniform(keys[0], minval=0.3, maxval=1.0, shape=(2,))
    tau_o = jrand.uniform(keys[1], minval=8, maxval=20)
    tau_r = jrand.uniform(keys[2], minval=3, maxval=12)
    g = 1.0

    return O_inf, R_inf, tau_o, tau_r, g


def _sample_scales(key, min_pc_fraction, max_pc_fraction,
                   num_traces, min_pc_scale, max_pc_scale):
    # sample scale values for photocurrents.
    # Randomly set some traces to have no photocurrent
    # according to pc_fraction.

    pc_fraction = jrand.uniform(key, minval=min_pc_fraction,
                                maxval=max_pc_fraction)
    key = jrand.fold_in(key, 0)
    pc_mask = jnp.where(
        jrand.uniform(key, shape=(num_traces,)) <= pc_fraction,
        1.0,
        0.0)
    key = jrand.fold_in(key, 0)
    pc_scales = jrand.uniform(key, shape=(num_traces,),
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
    observations = prev_pcs + cur_pcs + next_pcs + iid_noise + gp_noise

    return observations, targets


def gen_photocurrent_data(
    key,
    trial_dur=900,
    num_expts=1000,
    min_traces_per_expt=100,
    max_traces_per_expt=1000,
    photocurrent_scale_min=0.05,
    photocurrent_scale_max=1.0,
    psc_scale_max=1.0,
    psc_scale_min=0.01,
    psc_generation_kwargs=None,
    dtype=np.float32,
    ):

    # Generate length (in traces) for each experiment
    exp_lengths = jrand.randint(
        key,
        minval=min_traces_per_expt,
        maxval=max_traces_per_expt,
        size=num_expts)

    # generate all psc traces from neural demixer.
    # This is faster than calling it separately many times.
    if psc_generation_kwargs is None:
        psc_generation_kwargs = dict()
    demixer = cm.NeuralDemixer()
    demixer.generate_training_data(
        size=np.sum(exp_lengths), training_fraction=1.0, **psc_generation_kwargs)
    pscs, _ = demixer.training_data

    # generate all photocurrent templates
    

    # For each experiment, we generate a group of traces that share a common photocurrent component.
    # Some random fraction of these will _not_ have photocurrent present.
    for i in tqdm.trange(num_expts):

        
    return expts