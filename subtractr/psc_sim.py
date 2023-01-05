import jax.numpy as jnp
import jax.random as jrand
import subtractr.photocurrent_sim as photocurrent_sim

from jax import vmap, jit
from functools import partial

def _psc_kernel(tau_r, tau_d, delta, x):
    return (jnp.exp(-(x - delta)/tau_d) - jnp.exp(-(x - delta)/tau_r)) * (x >= delta)
psc_kernel_batched = vmap(_psc_kernel, in_axes=(0, 0, 0, None))


@partial(jit, static_argnames=('trial_dur', 'max_samples'))
def _sample_psc_kernel(key, trial_dur=900, tau_r_lower=10, tau_r_upper=80, tau_diff_lower=50,
                       tau_diff_upper=150, delta_lower=100, delta_upper=200, max_samples=4,
                       n_samples_active=1, amplitude_lower=0.01, amplitude_upper=0.5):
    """Sample PSCs with random time constants, onset times, and amplitudes."""
    keys = iter(jrand.split(key, num=4))
    tau_r_samples = jrand.uniform(
        next(keys), minval=tau_r_lower, maxval=tau_r_upper, shape=(max_samples,))
    tau_diff_samples = jrand.uniform(
        next(keys), minval=tau_diff_lower, maxval=tau_diff_upper, shape=(max_samples,))
    tau_d_samples = tau_r_samples + tau_diff_samples
    delta_samples = jrand.uniform(
        next(keys), minval=delta_lower, maxval=delta_upper, shape=(max_samples,))
    xeval = jnp.arange(trial_dur)

    pscs = psc_kernel_batched(
        tau_r_samples, tau_d_samples, delta_samples, xeval)

    max_vec = jnp.max(pscs, axis=-1, keepdims=True)
    amplitude = jrand.uniform(next(keys), minval=amplitude_lower,
                              maxval=amplitude_upper, shape=(max_samples, 1))

    # zero out samples beyond n_samples_active. This way, this function
    # always returns a constant shaped output
    active = jnp.arange(max_samples)[:, None] < n_samples_active
    amplitude *= active


    return pscs/max_vec * amplitude


@partial(jit, static_argnames=('trial_dur', 'mode_probs', 'prev_mode_probs', 'next_mode_probs',))
def _sample_pscs_single_trace(key, trial_dur=900, size=1000, training_fraction=0.9, lp_cutoff=500,
                              srate=20000, tau_r_lower=10, tau_r_upper=80, tau_diff_lower=2, tau_diff_upper=150,
                              delta_lower=160, delta_upper=400, next_delta_lower=400, next_delta_upper=899,
                              prev_delta_lower=-400, prev_delta_upper=-100,
                              amplitude_lower=0.01, amplitude_upper=0.5,
                              mode_probs=None, prev_mode_probs=None, next_mode_probs=None,
                              max_modes=4):

    if mode_probs is None:
        mode_probs = jnp.array([0.4, 0.4, 0.1, 0.1])
    else:
        mode_probs = jnp.array(mode_probs)
    if prev_mode_probs is None:
        prev_mode_probs = jnp.array([0.5, 0.4, 0.05, 0.05])
    else:
        prev_mode_probs = jnp.array(prev_mode_probs)
    if next_mode_probs is None:
        next_mode_probs = jnp.array([0.5, 0.4, 0.05, 0.05])
    else:
        next_mode_probs = jnp.array(next_mode_probs)
    

    keys = iter(jrand.split(key, num=10))
    n_modes = jrand.choice(next(keys), max_modes, shape=(1,), p=mode_probs)
    n_modes_prev = jrand.choice(
        next(keys), max_modes, shape=(1,), p=prev_mode_probs)
    n_modes_next = jrand.choice(
        next(keys), max_modes, shape=(1,), p=next_mode_probs)

    max_samples = len(prev_mode_probs)
    target = jnp.sum(_sample_psc_kernel(next(keys), trial_dur=trial_dur, tau_r_lower=tau_r_lower,
                                        tau_r_upper=tau_r_upper, tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper,
                                        delta_lower=delta_lower, delta_upper=delta_upper, n_samples_active=n_modes,
                                        amplitude_lower=amplitude_lower, amplitude_upper=amplitude_upper), axis=0)


    prev_psc = jnp.sum(_sample_psc_kernel(next(keys), trial_dur=trial_dur, tau_r_lower=tau_r_lower,
                                          tau_r_upper=tau_r_upper, tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper,
                                          delta_lower=prev_delta_lower, delta_upper=prev_delta_upper, n_samples_active=n_modes_prev,
                                          amplitude_lower=amplitude_lower, amplitude_upper=amplitude_upper), axis=0)

    next_psc = jnp.sum(_sample_psc_kernel(next(keys), trial_dur=trial_dur, tau_r_lower=tau_r_lower,
                                          tau_r_upper=tau_r_upper, tau_diff_lower=tau_diff_lower, tau_diff_upper=tau_diff_upper,
                                          delta_lower=next_delta_lower, delta_upper=next_delta_upper, n_samples_active=n_modes_next,
                                          amplitude_lower=amplitude_lower, amplitude_upper=amplitude_upper), axis=0)

    
    inputs = prev_psc + target + next_psc

    return inputs, target


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    key = jrand.PRNGKey(0)

    # Params
    tau_r_lower = 10
    tau_r_upper = 40

    # Params for chrome2f + interneuron -> pyramidal currents
    # tau_diff_lower = 150
    # tau_diff_upper = 340
    # convolve = False
    # sigma = 1

    # Params for chrome1 + pyramidal -> pyramidal currents
    tau_diff_lower = 60
    tau_diff_upper = 120

    num_samples = 20
    keys = jrand.split(key, num=20)

    # fold in args before vmap
    _sample_pscs_partial = partial(_sample_pscs_single_trace, trial_dur=900, delta_lower=160,
                                       delta_upper=400, next_delta_lower=400, next_delta_upper=899,
                                       prev_delta_upper=150, tau_diff_lower=tau_diff_lower,
                                       tau_diff_upper=tau_diff_upper, tau_r_lower=tau_r_lower,
                                       tau_r_upper=tau_r_upper,)
    sample_pscs_batch = vmap(_sample_pscs_partial)

    inputs, targets = sample_pscs_batch(keys)

    plt.plot(inputs.T)
    plt.plot(targets.T)
    plt.show()