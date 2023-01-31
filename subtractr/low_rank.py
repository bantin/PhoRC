import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial

def _svd_init(traces):
    U, S, V = jnp.linalg.svd(traces)
    U = U[:, 0:1] * S[0]
    V = V[0:1, :]
    U, V = jax.lax.cond(
        jnp.all(jnp.array([jnp.all(U <= 1e-10), jnp.all(V <= 1e-10)])),
        lambda _: (U * -1, V * -1),
        lambda _: (U, V),
        None
    )
    return U, V
    

def _rank_one_nmu(traces, init_factors,
    update_U=True, update_V=True, 
    baseline=False, stim_start=100, 
    maxiter=5000, tol=1e-2, rho=2.0,):
    """Non-negative matrix underapproximation with rank 1 using ADMM

    Init factors must be passed as a tuple of U and V matrices, due to the way JAX handles
    optional arguments. If init_factors_contains NaN, override with SVD initialization.

    Parameters
    ----------
    traces : array-like
        Traces to estimate photocurrents from. Shape is (n_traces, n_timepoints).
    init_factors : tuple of array-like
        Initial U and V matrices. Shape is (n_traces, rank) and (rank, n_timepoints).
    update_U : bool
        If True, update U.
    update_V : bool
        If True, update V.
    maxiter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    rho : float
        ADMM penalty parameter.
    init_factors : tuple of array-like
        Initial U and V matrices. Shape is (n_traces, rank) and (rank, n_timepoints).
        If None, initialize using SVD.

    Returns
    -------
    U : array-like
        U matrix. Shape is (n_traces, rank).
    V : array-like
        V matrix. Shape is (rank, n_timepoints).
    beta : array-like
        Baseline term. Shape is (n_traces, 1).
    max_violation : array-like
        Maximum violation of non-underapprox constraint at each iteration.
    """  
    U, V = init_factors
    U, V = jax.lax.cond(
        jnp.any(jnp.isnan(U @ V)),
        lambda _: _svd_init(traces),
        lambda _: init_factors,
        None
    )

    R = jnp.maximum(0, traces - U @ V)
    Gamma = jnp.zeros_like(traces)
    beta = jnp.zeros((traces.shape[0], 1)) # baseline term
    if baseline:
        beta = jnp.mean(traces, axis=-1, keepdims=True)
    U_old = U.copy() + 10 * tol
    V_old = V.copy() + 10 * tol
    k = 0
    max_violation = jnp.zeros(maxiter) * jnp.nan
    val = (U, V, U_old, V_old, R, Gamma, k, max_violation, beta) 


    def _convergence_tol_check(val):
        U, V, U_old, V_old, R, Gamma, k, max_violation, beta = val
        return jnp.any(jnp.array([
            (jnp.linalg.norm(U - U_old) / jnp.linalg.norm(U_old) > tol),
            (jnp.linalg.norm(V - V_old) / jnp.linalg.norm(V_old) > tol),
            jnp.min(traces - U @ V - beta) < -tol,
        ]))

    def _not_converged(val):
        U, V, U_old, V_old, R, Gamma, k, max_violation, beta = val
        return jax.lax.cond(
            k < maxiter,
            lambda _: _convergence_tol_check(val),
            lambda _: False,
            None
        )

    def _update(val):
        U, V, U_old, V_old, R, Gamma, k, max_violation, beta = val
        U_old = U.copy()
        V_old = V.copy()
        M = traces - R - beta + Gamma / rho
        resid = traces - U @ V - beta
        if update_U:
            U = M @ V.T
            U = jnp.maximum(0, U)
            U = jax.lax.cond(
                jnp.all(U <= 1e-10),
                lambda _: U * 0.0,
                lambda _: U / jnp.linalg.norm(U),
                None,
            )
        if update_V:
            V = (U.T @ M) / (U.T @ U + 1e-10)
            V = jnp.maximum(0, V)
        if baseline:
            beta = jnp.sum(1 / rho * Gamma + traces - U @ V - R, axis=-1, keepdims=True) / traces.shape[1]
            beta = jnp.maximum(0, beta)

        R = 1 / (1 + rho) * (rho * resid + Gamma)
        R = jnp.maximum(0, R)
        Gamma = Gamma + rho * (resid - R)
        max_violation = max_violation.at[k].set(jnp.min(traces - U @ V - beta))
        k += 1
        return (U, V, U_old, V_old, R, Gamma, k, max_violation, beta)
    val = jax.lax.while_loop(
        _not_converged,
        _update,
        val,
    )
    (U, V, U_old, V_old, R, Gamma, k, max_violation, beta) = val

    return U, V, beta, max_violation


# @partial(jit, static_argnames=('update_U', 'update_V'))
def _nmu(traces, init_factors, update_U=True, update_V=True,
        rank=1, baseline=False, stim_start=100, **kwargs):
    """Non-negative matrix underapproximation, solved recursively by updating rank-1 factors.

    Parameters
    ----------
    traces : array-like
        Traces to estimate photocurrents from. Shape is (n_traces, n_timepoints).
    init_factors : tuple of array-like
        Initial U and V matrices. Shape is (n_traces, rank) and (rank, n_timepoints).
        Pass in all NaN to use SVD init in the inner loop.
    rank : int
        Rank of the estimated matrix.
    update_U : bool
        If True, update U.
    update_V : bool 
        If True, update V.
    init_factors : tuple of array-like
        Initial U and V matrices. Shape is (n_traces, rank) and (rank, n_timepoints).
    kwargs : dict
        This kwargs dict can contain arguments for the ADMM step, including:
        maxiter : int, default 100
            Maximum number of iterations.
        tol : float, default 1e-3
            Tolerance for convergence.
        rho : float, default 1e-3 
            ADMM penalty parameter.

    Returns
    -------
    U : array-like
        Estimated U matrix. Shape is (n_traces, rank).
    V : array-like
        Estimated V matrix. Shape is (rank, n_timepoints).
    beta : array-like
        Estimated baseline term. Shape is (n_traces, 1).
    """
    U, V = init_factors
    N, rank = U.shape

    # run the first component separately to possibly allow for a baseline term
    U_0, V_0, beta, _, = _rank_one_nmu(
        traces,
        (U[:, 0:1], V[0:1, :]),
        update_U=update_U,
        update_V=update_V,
        baseline=baseline,
        stim_start=stim_start,
        **kwargs,
    )   
    traces = traces - U_0 @ V_0 - beta

    # The call to jax.lax.scan is equivalent to this:
    # for i in range(rank):
    #     U_i, V_i, _ = _rank_one_nmu(
    #         traces,
    #         (U[:, i:i+1], V[i:i+1, :]),
    #         update_U=update_U,
    #         update_V=update_V,
    #         **admm_kwargs,
    #     )
    #     U = U.at[:, i:i+1].set(U_i)
    #     V = V.at[i:i+1, :].set(V_i)
    #     traces = traces - U_i @ V_i

    def _scan_inner(val, curr_init_factors):
        traces = val
        U_i_init, V_i_init = curr_init_factors
        U_i_init = U_i_init[:, None]
        V_i_init = V_i_init[None, :]
        U_i, V_i, _, _, = _rank_one_nmu(
            traces,
            (U_i_init, V_i_init),
            update_U=update_U,
            update_V=update_V,
            baseline=False,
            **kwargs,
        )
        traces = traces - U_i @ V_i
        return traces, (jnp.squeeze(U_i), jnp.squeeze(V_i))
    _, (U_transpose, V) = jax.lax.scan(
        _scan_inner,
        traces,
        (U.T[1:], V[1:]),
    )
    U = jnp.concatenate((U_0, U_transpose.T), axis=1)
    V = jnp.concatenate((V_0, V), axis=0)
    return U, V, beta

@partial(jit, static_argnames=('constrain_V', 'rank', 'stim_start', 'stim_end', 'baseline'))
def estimate_photocurrents_nmu(traces, 
    stim_start=100, stim_end=200, constrain_V=False, rank=1, baseline=False):
    """Estimate photocurrents using non-negative matrix underapproximation.

    Parameters
    ----------
    traces : array-like
        Traces to estimate photocurrents from. Shape is (n_traces, n_timepoints).
    stim_start : int
        Index of first timepoint of stimulus.
    stim_end : int
        Index of last timepoint of stimulus.
    constrain_V : bool
        If True, constrain the estimated V using the underapprox constraint.
    rank : int
        Rank of the estimated matrix.

    Returns
    -------
    U : array-like
        Estimated U matrix. Shape is (n_traces, rank).
    V : array-like
        Estimated V matrix. Shape is (rank, n_timepoints).
    beta : array-like
        Estimated baseline term. Shape is (n_traces, 1).
    """
    # Create dummy initial factors
    # to use SVD initialization inside _rank_one_nmu
    U_init = jnp.zeros((traces.shape[0], rank)) * jnp.nan
    traces = jnp.maximum(0, traces)
    if baseline:
        start_idx = 0
    else:
        start_idx = stim_start
    V_init = jnp.zeros((rank, stim_end - start_idx)) * jnp.nan
    
    U_stim, V_stim, beta = _nmu(traces[:, start_idx:stim_end], (U_init, V_init), 
        rank=rank, update_U=True, update_V=True, baseline=baseline, stim_start=stim_start)
    
    V_full = jnp.linalg.lstsq(U_stim, traces[:, stim_start:])[0]
    if constrain_V:
        _, V_full, _ = _nmu(traces[:, stim_start:], rank=rank, init_factors=(U_stim, V_full),
            update_U=False, update_V=True, baseline=False)
    # pad V with zeros to account for the time before stim_start
    V_full = jnp.concatenate((jnp.zeros((rank, stim_start)), V_full), axis=1)
    return U_stim, V_full, beta

def estimate_photocurrents_by_batches(traces,
                           rank=1, constrain_V=True, baseline=False,
                           stim_start=100, stim_end=200, batch_size=-1,
                           subtract_baseline=True):
    """Estimate photocurrents using non-negative matrix underapproximation.

    Parameters
    ----------
    traces : array-like
        Traces to estimate photocurrents from. Shape is (n_traces, n_timepoints).
    rank : int
        Rank of the estimated matrix.
    constrain_V : bool
        If True, constrain the estimated V using the underapprox constraint.
    baseline : bool
        If True, estimate a baseline term.
    stim_start : int
        Index of first timepoint of stimulus.
    stim_end : int
        Index of last timepoint of stimulus.
    batch_size : int
        Number of traces to process in each batch. If -1, process all traces
        at once.
    subtract_baseline : bool
        If True, subtract the baseline from the traces in estimate
    
    Returns
    -------
    U : array-like
        Estimated U matrix. Shape is (n_traces, rank).
    V : array-like
        Estimated V matrix. Shape is (rank, n_timepoints).
    beta : array-like
        Estimated baseline term. Shape is (n_traces, 1).
    """
    def _make_estimate(pscs, stim_start=100, stim_end=200):
        U, V, beta = estimate_photocurrents_nmu(
            pscs,
            rank=rank,
            stim_start=stim_start,
            stim_end=stim_end,
            constrain_V=constrain_V,
            baseline=baseline,
        )
        return U @ V + beta

    # sort traces by magnitude around stim
    idxs = np.argsort(np.linalg.norm(
        traces[:, stim_start:stim_end+50], axis=-1))

    # save this so that we can return estimates in the original (unsorted) order
    reverse_idxs = np.argsort(idxs)
    traces = traces[idxs]

    if batch_size == -1:
        est = _make_estimate(
            traces, stim_start, stim_end
        )
    else:
        est = np.zeros_like(traces)
        num_complete_batches = traces.shape[0] // batch_size
        max_index = num_complete_batches * batch_size
        folded_traces = traces[:max_index].reshape(
            num_complete_batches, batch_size, traces.shape[1])
        
        # take advantage of vmap to run in parallel on all batches (except the last one)
        # ests_batched = _make_estimate_batched(folded_traces, stim_start, stim_end)
        # print('got here')
        est[:max_index] = np.concatenate(
            [_make_estimate(x, stim_start, stim_end) for x in folded_traces], axis=0)

        # re-run on last batch, in case the number of traces is not divisible by the batch size
        if traces.shape[0] % batch_size != 0:
            est[-batch_size:] = _make_estimate(traces[-batch_size:],
                                               stim_start, stim_end)

    est = est[reverse_idxs]
    return est