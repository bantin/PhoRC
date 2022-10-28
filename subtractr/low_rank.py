from posixpath import sep
import jax.numpy as jnp
from jax import jit, vmap
from subtractr.photocurrent_sim import monotone_decay_filter

import subtractr.nmu as nmu
import numpy as np



def _scalar_underapprox(x, u):
    # NEED TO FIX
    alpha_hat = jnp.linalg.norm(x) / (jnp.linalg.norm(u) + 1e-16)
    val = jnp.all(alpha_hat * u <= x).astype(int)
    max_viol_idx = jnp.argmax(u)
    return  val * alpha_hat + (1 - val) * x[max_viol_idx] / u[max_viol_idx]

scalar_underapprox = jit(vmap(_scalar_underapprox, in_axes=(1,None)))

# import cvxpy as cp

# def _scalar_underapprox(x, u):
#     alpha_cvx = cp.Variable()
#     prob = cp.Problem(cp.Minimize(cp.sum_squares(x - alpha_cvx * u)),
#                     [alpha_cvx * u <= x,
#                     ])
#     prob.solve()
#     return alpha_cvx.value

# def scalar_underapprox(X, U):
#     import pdb; pdb.set_trace()
#     return np.array([_scalar_underapprox(X[:,i],U) for i in range(X.shape[-1])])


def _nmu_factors_to_matrices(factors):
    U = np.concatenate([factor[0] for factor in factors], axis=-1)
    V = np.concatenate([factor[1] for factor in factors], axis=0)
    return (U, V)

def _nmu_matrices_to_factors(U, V):
    return [(u_n[:, None], v_t[None, :]) for (u_n, v_t) in zip(U.T, V)]

def _nmu_estimate_with_baseline(pscs, stim_start=100, stim_end=200):
    pscs_truncated = np.maximum(pscs, 0)


    # Initialize U_stim by fitting on part of the data matrix
    # where the laser is on. 
    U_stim, V_stim, mu  = nmu.nmu_admm_with_baseline(pscs_truncated[:, 0:stim_end])

    # subtract baseline off of the pscs
    pscs_truncated -= mu

    V_post = np.linalg.lstsq(U_stim, pscs_truncated)[0]

    # run temporal waveform through monotone decay filter
    # to avoid picking up effects from the following trial
    V_post = monotone_decay_filter(V_post)
    V_post[0:stim_start] = 0.0

    # V_post = np.linalg.lstsq(U_stim, pscs_truncated[:,stim_end:])[0]
    # V_final = np.zeros((1, pscs_truncated.shape[-1]))
    # V_final[:,0:stim_end] = V_stim
    # V_final[:,stim_end:] = V_post
    # return U_stim, V_final
    return U_stim, V_post

def _nmu_estimate_stepwise(pscs, rank=1, init_factors=None,
        stim_start=100, stim_end=200, stepwise_constrain_V=True):
    pscs_truncated = np.maximum(pscs, 1e-8)


    # Initialize U_stim by fitting on part of the data matrix
    # where the laser is on. 
    init_factors_small = None
    if init_factors is not None:
        init_factors_small = [(x, y[:, stim_start:stim_end]) for 
            x,y in init_factors]
    stim_factors  = nmu.recursive_nmu(pscs_truncated[:, stim_start:stim_end],
        r=rank, init_factors=init_factors_small)
    U_stim, _ = _nmu_factors_to_matrices(stim_factors)

    # get the final V by fitting on the full dataset, holding U fixed,
    # with the underapprox constraint
    if stepwise_constrain_V:

        # If init_factors is provided, use the waveform shape that was passed.
        # Otherwise, run nmu on the full traces to get an initializing waveform
        if init_factors is None:
            full_factors = nmu.recursive_nmu(pscs_truncated, r=rank)
        
            # init factors gets U from stim_factors and V from full matrix
            init_factors = [(x[0], y[1]) for x,y in zip(stim_factors, full_factors)]
        else:
            # init factors gets U from stim_factors and v from the passed waveform
            init_factors = [(x[0], y[1]) for x,y in zip(stim_factors, init_factors)]

        final_factors = nmu.recursive_nmu(pscs_truncated,
            r=rank, init_factors=init_factors,
            update_u=False, update_v=True)
        _, V_final = _nmu_factors_to_matrices(final_factors)

    # no underapprox constraint, just solve a least squares problem
    else:
        V_final = np.linalg.lstsq(U_stim, pscs_truncated)[0]


    return U_stim, V_final

def _nmu_estimate(pscs, rank=1):
    pscs_truncated = np.maximum(pscs, 1e-8)
    factors  = nmu.recursive_nmu(pscs_truncated, r=rank)
    U, V = _nmu_factors_to_matrices(factors)
    return U, V

def estimate_photocurrents(pscs, I, rank=1,
        separate_by_power=True, stepwise=False,
        stepwise_constrain_V=True):
    if separate_by_power:
        est = np.zeros_like(pscs)
        powers = np.unique(I)

        # Iterate over powers, using waveform from previous power for initialization
        V_prev = None
        for power in powers:
            these_trials = (I == power)

            # if V_prev is not none, form the least squares estimate of U, then
            # pass that as the initialization to NMU
            init_factors = None
            if V_prev is not None:
                U_init = np.maximum(np.linalg.lstsq(V_prev.T, pscs[these_trials,:].T, rcond=0.01)[0].T, 0)
                init_factors = _nmu_matrices_to_factors(U_init, V_prev)
            if stepwise:
                U, V = _nmu_estimate_stepwise(pscs[these_trials],
                    init_factors=init_factors,
                    rank=rank, stepwise_constrain_V=stepwise_constrain_V)
            else:
                U, V = _nmu_estimate(pscs[these_trials],
                    rank=rank)
            est[these_trials] = U @ V
            V_prev = V
        return est
    else:
        if stepwise:
            U, V = _nmu_estimate_stepwise(pscs,
                rank=rank, stepwise_constrain_V=stepwise_constrain_V)
        else:
            U, V = _nmu_estimate(pscs, rank=rank)
        return U @ V

    return U, V

def estimate_photocurrents_baseline(pscs, I, rank=1,
        separate_by_power=True, stim_start=100, stim_end=200,
        stepwise_constrain_V=True):
    assert rank == 1, "Only rank-1 implemented with baseline"

    if separate_by_power:
        est = np.zeros_like(pscs)
        powers = np.unique(I)
        for power in powers:
            these_trials = (I == power)
            U, V = _nmu_estimate_with_baseline(pscs[these_trials],
                stim_start=stim_start,
                stim_end=stim_end)
            est[these_trials] = U @ V
        return est
    else:
        U, V = _nmu_estimate_with_baseline(pscs,
            stim_start=stim_start,
            stim_end=stim_end
            )
        est = U @ V
        return est
