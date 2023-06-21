import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial
from subtractr.pava import pava_decreasing
from tqdm import tqdm
from collections import namedtuple

# Define a photocurrent estimate, which contains estimates for previous trial,
# as well as baseline
PhotocurrentEstimate = namedtuple(
    'PhotocurrentEstimate',
    ['U_pre', 'V_pre', 'U_photo', 'V_photo', 'beta', 'fit_info']
)


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
                  maxiter=100, tol=1e-2, rho=2.0,):
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

    Gamma = jnp.zeros_like(traces)
    beta = jnp.zeros((traces.shape[0], 1))  # baseline term
    if baseline:
        beta = jnp.mean(traces, axis=-1, keepdims=True)
        V = V.at[:, 0:stim_start].set(0)
    R = jnp.maximum(0, traces - U @ V)
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
                lambda _: U,
                None,
            )
        if update_V:
            V = (U.T @ M) / (U.T @ U + 1e-10)
            V = jnp.maximum(0, V)
            V = jax.lax.cond(
                jnp.all(V <= 1e-10),
                lambda _: V * 0.0,
                lambda _: V / jnp.linalg.norm(V),
                None,
            )
        if baseline:
            V = V.at[:, 0:stim_start].set(0)
            beta = jnp.sum(1 / rho * Gamma + traces - U @ V - R,
                           axis=-1, keepdims=True) / traces.shape[1]
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


def _rank_one_nmu_decreasing(traces, init_factors,
                             update_U=True, update_V=True,
                             dec_start=0,  # index where we enforce the constraint
                             maxiter=500, tol=1e-2, rho=1.0, gamma=1.0):
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

    Gamma = jnp.zeros_like(traces)
    lam = jnp.zeros_like(V)
    q = V.copy()
    R = jnp.maximum(0, traces - U @ V)
    U_old = U.copy() + 10 * tol
    V_old = V.copy() + 10 * tol
    k = 0
    max_violation = jnp.zeros(maxiter) * jnp.nan
    val = (U, V, q, lam, U_old, V_old, R, Gamma, k, max_violation)

    # NB: Beta is a placeholder for now -- not actually implemented.
    # This allows us to have the same function signature as _rank_one_nmu.
    beta = jnp.zeros((traces.shape[0], 1))

    def _convergence_tol_check(val):
        U, V, q, lam, U_old, V_old, R, Gamma, k, max_violation = val
        return jnp.any(jnp.array([
            (jnp.linalg.norm(U - U_old) / jnp.linalg.norm(U_old) > tol),
            (jnp.linalg.norm(V - V_old) / jnp.linalg.norm(V_old) > tol),
            (jnp.linalg.norm(V - q) > tol),
            jnp.min(traces - U @ V) < -tol,
        ]))

    def _not_converged(val):
        U, V, q, lam, U_old, V_old, R, Gamma, k, max_violation = val
        return jax.lax.cond(
            k < maxiter,
            lambda _: _convergence_tol_check(val),
            lambda _: False,
            None
        )

    def _update(val):
        U, V, q, lam, U_old, V_old, R, Gamma, k, max_violation = val
        U_old = U.copy()
        V_old = V.copy()
        M = traces - R + Gamma / rho
        resid = traces - U @ V
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

            # update V
            V = (U.T @ (Gamma + rho * (traces - R)) +
                 lam + rho * q) / (rho * (U.T @ U + 1))
            V = jnp.maximum(0, V)

            # update auxiliary variable q
            q = jnp.zeros_like(V)
            q = q.at[0, 0:dec_start].set((V - lam / rho)[0, 0:dec_start])
            q = q.at[0, dec_start:].set(
                pava_decreasing(jnp.squeeze(V - lam / rho)
                                [dec_start:], gamma=gamma)
            )

            # update lagrange multiplier
            lam = lam + rho * (q - V)

        R = 1 / (1 + rho) * (rho * resid + Gamma)
        R = jnp.maximum(0, R)
        Gamma = Gamma + rho * (resid - R)
        max_violation = max_violation.at[k].set(jnp.min(traces - U @ V))
        k += 1
        return (U, V, q, lam, U_old, V_old, R, Gamma, k, max_violation)
    val = jax.lax.while_loop(
        _not_converged,
        _update,
        val,
    )
    (U, V, q, lam, U_old, V_old, R, Gamma, k, max_violation) = val

    return U, V, beta, max_violation


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


@partial(jit, static_argnames=('stim_start', 'stim_end', 'rank'), backend='cpu')
def estimate_photocurrents_nmu_extended_baseline(traces,
                                                 stim_start=100, stim_end=200, rank=1, gamma=0.999):
    """Estimate photocurrents using non-negative matrix underapproximation.

    Parameters
    ----------
    traces : array-like
        Traces to estimate photocurrents from. Shape is (n_traces, n_timepoints).
    stim_start : int
        Index of first timepoint of stimulus.
    stim_end : int
        Index of last timepoint of stimulus.
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
    start_idx = 0
    V_init = jnp.zeros((rank, stim_end - start_idx)) * jnp.nan

    # Fit baseline along with first rank-one term
    U_stim, V_stim, beta = _nmu(traces[:, start_idx:stim_end], (U_init, V_init),
                                rank=rank, update_U=True, update_V=True, baseline=True, stim_start=stim_start)

    # Use beta as the weights in U to fit a rank-one term
    # to the entire trace, constraining V to be decreasing
    U_baseline_init = beta
    V_baseline_init = jnp.ones((1, traces.shape[1]))
    baseline_init_factors = (U_baseline_init, V_baseline_init)
    U_baseline, V_baseline, _ = _rank_one_nmu_decreasing(traces, baseline_init_factors,
                                                         update_U=False, update_V=True, gamma=gamma)
    traces = traces - U_baseline @ V_baseline

    # Fit the rest of V using NMU
    V_photo_init = jnp.linalg.lstsq(U_stim, traces[:, stim_start:])[0]
    _, V_photo, _ = _nmu(traces[:, stim_start:], rank=rank, init_factors=(U_stim, V_photo_init),
                         update_U=False, update_V=True, baseline=False)

    # pad V with zeros to account for the time before stim_start
    V_photo = jnp.concatenate((jnp.zeros((rank, stim_start)), V_photo), axis=1)

    # Create full U and V adding the learned baseline term
    U_full = jnp.concatenate((U_stim, U_baseline), axis=1)
    V_full = jnp.concatenate((V_photo, V_baseline), axis=0)

    # zero out beta since we already have it in U
    beta = jnp.zeros_like(beta)

    return U_full, V_full, beta


def estimate(traces,
                                      rank=1, constrain_V=True, baseline=False,
                                      stim_start=100, stim_end=200, batch_size=-1,
                                      subtract_baselines=True, method='coordinate_descent',
                                      **kwargs):
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
    estimator_dict = dict(
        naive_svd=estimate_photocurrents_naive_svd,
        coordinate_descent=estimate_photocurrents_nmu_coordinate_descent,
    )

    assert method in estimator_dict, f"method must be one of {list(estimator_dict.keys())}"
    estimator = estimator_dict[method]
    def _make_estimate(pscs, stim_start, stim_end):
        result = estimator(pscs, rank=rank,
            stim_start=stim_start, stim_end=stim_end,
            **kwargs)
        est = result.U_photo @ result.V_photo
        if subtract_baselines:
            est += result.U_pre @ result.V_pre + result.beta
        return est

    # sort traces by magnitude around stim
    traces = np.maximum(0, traces)
    idxs = np.argsort(np.linalg.norm(
        traces[:, stim_start:stim_end], axis=-1))[::-1]

    # save this so that we can return estimates in the original (unsorted) order
    reverse_idxs = np.argsort(idxs)
    traces = traces[idxs]

    if batch_size == -1:
        print('Running photocurrent estimation with no batching...')
        est = _make_estimate(
            traces, stim_start, stim_end
        )
    else:
        est = np.zeros_like(traces)
        num_complete_batches = traces.shape[0] // batch_size
        max_index = num_complete_batches * batch_size
        folded_traces = traces[:max_index].reshape(
            num_complete_batches, batch_size, traces.shape[1])

        print('Running photocurrent estimation with %d batches...' % num_complete_batches)
        est[:max_index] = np.concatenate(
            [_make_estimate(x, stim_start, stim_end) for x in tqdm(folded_traces)], axis=0)

        # re-run on last batch, in case the number of traces is not divisible by the batch size
        if traces.shape[0] % batch_size != 0:
            est[-batch_size:] = _make_estimate(traces[-batch_size:],
                                               stim_start, stim_end)

    est = est[reverse_idxs]
    return est



def _nonnegsvd_init(traces, rank=1):
    U, S, V = jnp.linalg.svd(traces, full_matrices=False)
    U = U[:, :rank] * S[:rank]
    V = V[:rank, :]

    U_plus = jnp.maximum(U, 0)
    U_minus = jnp.maximum(-U, 0)
    V_plus = jnp.maximum(V, 0)
    V_minus = jnp.maximum(-V, 0)

    U = jnp.zeros_like(U)
    V = jnp.zeros_like(V)
    for i in range(rank):
        # Rewrite the above using jax.lax.cond
        
        U = U.at[:, i:i+1].set(
            jax.lax.cond(
                jnp.linalg.norm(
                    U_plus[:, i:i+1] @ V_plus[i:i+1, :]) > jnp.linalg.norm(
                        U_minus[:, i:i+1] @ V_minus[i:i+1, :]),
                lambda x: U_plus[:, i:i+1],
                lambda x: U_minus[:, i:i+1],
                None
            )
        )
        V = V.at[i:+1, :].set(
            jax.lax.cond(
                jnp.linalg.norm(
                    U_plus[:, i:i+1] @ V_plus[i:i+1, :]) > jnp.linalg.norm(
                    U_minus[:, i:i+1] @ V_minus[i:i+1, :]),
                lambda x: V_plus[i:i+1, :],
                lambda x: V_minus[i:i+1, :],
                None
            )
        )

    return U, V


def coordinate_descent_nmu(traces,
                           stim_start=100,
                           const_baseline=True,
                           decaying_baseline=True,
                           rank=1,
                           update_U=True,
                           update_V=True,
                           max_iters=20,
                           tol=1e-4,
                           gamma=0.999,
                           rho=1e-2):
    traces = jnp.array(traces)
    def _subtract_photo(traces, U_photo, V_photo):
        traces = traces.copy()
        return traces.at[:, stim_start:].set(traces[:, stim_start:] - U_photo @ V_photo)

    def _update_factors(i, outer_state):

        # unpack outer state
        U_photo, V_photo, U_base, V_base, beta, loss = outer_state

        # update decaying baseline if active
        if decaying_baseline:
            resid = _subtract_photo(traces, U_photo, V_photo)
            U_base, V_base, _, _ = _rank_one_nmu_decreasing(resid, (U_base, V_base),
                                                             maxiter=500, rho=rho, gamma=gamma)

        # update constant baseline if active
        if const_baseline:
            resid = _subtract_photo(traces, U_photo, V_photo) - U_base @ V_base
            beta = jnp.min(resid, axis=-1, keepdims=True)
            beta = jnp.maximum(beta, 0)

        # update rank-r approximation of the photocurrent component, 
        # stored in U_photo and V_photo. Use lax.scan to update each
        # rank-one component in turn
        def _scan_inner(resid, curr_init_factors):
            U_i_init, V_i_init = curr_init_factors
            U_i_init = U_i_init[:, None]
            V_i_init = V_i_init[None, :]
            resid = resid.at[:, stim_start:].add(U_i_init @ V_i_init)
            U_i, V_i, _, _, = _rank_one_nmu(
                resid[:, stim_start:],
                (U_i_init, V_i_init),
                update_U=update_U,
                update_V=update_V,
                baseline=False,
            )
            resid = resid.at[:, stim_start:].add(-U_i @ V_i)
            return resid, (jnp.squeeze(U_i), jnp.squeeze(V_i))

        # resid = traces - U_photo @ V_photo - U_base @ V_base - beta
        resid = _subtract_photo(traces, U_photo, V_photo) - U_base @ V_base - beta
        _, (U_photo_T, V_photo) = jax.lax.scan(
            _scan_inner,
            resid,
            (U_photo.T, V_photo),
        )
        U_photo = U_photo_T.T
        loss = loss.at[i].set(
            jnp.linalg.norm(_subtract_photo(traces, U_photo, V_photo) - U_base @ V_base - beta)
        )
        return U_photo, V_photo, U_base, V_base, beta, loss


    
    # Initialize U, V, and beta using period before stim for initializing the
    # baseline terms
    if decaying_baseline:
        U_pre, V_pre = _nonnegsvd_init(traces[:, 0:stim_start], rank=1)
        V_pre_init = jnp.linalg.lstsq(U_pre, traces)[0]
        V_pre = pava_decreasing(
            jnp.squeeze(V_pre_init), gamma=gamma)[None, :]
        
    else:
        U_pre, V_pre = jnp.zeros((traces.shape[0], 1)), jnp.zeros((1, traces.shape[1]))

    # U_photo, V_photo = _nonnegsvd_init(
    #     jnp.maximum(traces - U_pre @ V_pre, 0), rank=rank
    # )
    U_photo, _, _ = _nmu(jnp.maximum(traces - U_pre @ V_pre, 0)[:, stim_start:],
        (jnp.zeros((traces.shape[0], rank)) + jnp.nan, jnp.zeros((rank, traces.shape[1] - stim_start)) + jnp.nan),
        rank=rank, update_U=True, update_V=True, baseline=False, stim_start=stim_start)
    V_photo = jnp.linalg.lstsq(U_photo, traces[:, stim_start:])[0]
    
    beta = jnp.zeros((traces.shape[0], 1))

    loss = jnp.zeros(max_iters)
    outer_state = (U_photo, V_photo, U_pre, V_pre, beta, loss)
    outer_state = jax.lax.fori_loop(0, max_iters, _update_factors, outer_state)
    U_photo, V_photo, U_pre, V_pre, beta, loss = outer_state
    V_photo = jnp.concatenate((jnp.zeros((rank, stim_start)), V_photo), axis=1)

    result = PhotocurrentEstimate(U_pre=U_pre, V_pre=V_pre,
        U_photo=U_photo, V_photo=V_photo, beta=beta, fit_info=loss)
    return result

@partial(jit, static_argnames=('stim_start', 'stim_end', 'dec_start', 'gamma', 'rank', 'coordinate_descent_iters'), backend='cpu')
def estimate_photocurrents_nmu_coordinate_descent(traces,
                               stim_start=100, stim_end=200, dec_start=500,
                               gamma=0.999, rank=1, coordinate_descent_iters=5,
                               nmu_max_iters=1000, rho=0.01,):
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
    traces = jnp.maximum(0, traces)

    # Fit 3 terms to the very beginning of the matrix:
    # decaying baseline, constant baseline, and photocurrent
    result = coordinate_descent_nmu(
        traces[:,0:stim_end],
        const_baseline=True,
        decaying_baseline=False,
        rank=rank,
        max_iters=coordinate_descent_iters,
        rho=rho,
        gamma=gamma,
    )

    
    # Fit decaying baseline term to account for previous trial
    traces = traces - result.beta
    V_dec_full_init = jnp.linalg.lstsq(result.U_pre, traces)[0]
    _, V_pre, _, _ = _rank_one_nmu_decreasing(traces,
                        init_factors=(result.U_pre, V_dec_full_init),
                        update_U=False, update_V=True,
                        dec_start=0, gamma=gamma, rho=rho)
    # subtract away decaying baseline
    traces = traces - result.U_pre @ V_pre
    
    # Now fit photocurrent and output estimate.
    # We force our photocurrent estimate to be decreasing after dec_start
    V_photo = jnp.zeros((rank, traces.shape[1] - stim_start))
    for r in range(rank):
        u_curr = result.U_photo[:, r:r+1]
        v_photo_init = jnp.linalg.lstsq(u_curr, traces[:, stim_start:])[0]
        _, v_photo, _, _ = _rank_one_nmu_decreasing(traces[:, stim_start:],
                            init_factors=(u_curr, v_photo_init),
                            update_U=False, update_V=True, dec_start=dec_start,
                            gamma=gamma, rho=rho, maxiter=nmu_max_iters)
        V_photo = V_photo.at[r:r+1, :].set(v_photo)

        # subtract away photocurrent after stim start
        traces = traces.at[:, stim_start:].set(traces[:, stim_start:] - u_curr @ v_photo)
        
    # pad V with zeros to account for the time before stim_start
    V_photo = jnp.concatenate((jnp.zeros((rank, stim_start)), V_photo), axis=1)

    result = PhotocurrentEstimate(U_pre=result.U_pre, V_pre=V_pre,
        U_photo=result.U_photo, V_photo=V_photo, beta=result.beta, fit_info=result.fit_info)

    return result


def estimate_photocurrents_naive_svd(traces, rank=1):
    beta = jnp.zeros((traces.shape[0], 1))
    U, S, V = jnp.linalg.svd(traces, full_matrices=False)
    return U[:, :rank], jnp.diag(S[:rank]) @ V[:rank, :], beta