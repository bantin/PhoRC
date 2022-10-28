from tkinter import W
import numpy as np
import scipy.sparse.linalg as sp_linalg
from six import string_types

# copied from https://github.com/marianotepper/nmu_rfit
# Associated paper "Nonnegative Matrix Underapproximation for Robust Multiple Model Fitting"
# by Tepper and Sapiro

def recursive_nmu(array, r=None, max_iter=5e2, tol=1e-3, downdate='minus',
                  init_strategy='svd', init_factors=None,
                  update_u=True, update_v=True):
    if r is None:
        r = min(array.shape)

    array = array.copy()
    factors = []
    for k in range(r):
        if init_factors is not None:
            curr_init_factor = init_factors[k]
        else:
            curr_init_factor = None

        u, v = nmu_admm(array, max_iter, tol,
            init_strategy=init_strategy, update_u=update_u, update_v=update_v,
            init_factors=curr_init_factor)
        if np.count_nonzero(u) == 0 or np.count_nonzero(v) == 0:
            break
        factors.append((u, v))
        if k == r - 1:
            continue
        if downdate == 'minus':
            array = np.maximum(0, array - np.dot(u, v))
        if downdate == 'hard-col' or downdate == 'hard-both':
            array[:, np.squeeze(v > 0)] = 0
        if downdate == 'hard-row' or downdate == 'hard-both':
            array[np.squeeze(u > 0), :] = 0
        if array.max() == 0:
            break

    return factors


def nmu(array, max_iter=5e2, tol=1e-3, init='svd', ret_errors=False):
    u, v = _nmu_initialize(array, init=init)
    u_old = u.copy()
    v_old = v.copy()
    mu = 0

    # Alternating optimization
    error_u = []
    error_v = []
    for k in range(int(max_iter)):
        # updating mu:
        if np.any(u > 0) and np.any(v > 0):
            remainder = array - u.dot(v)
            mu = np.maximum(0, mu - remainder / (k + 1))
        else:
            mu /= 2
            u = u_old
            v = v_old

        u_old = u.copy()
        v_old = v.copy()
        # updating u, v:
        aux = array - mu
        u = np.maximum(0, aux.dot(v.T))
        u = np.maximum(0, u)
        umax = u.max()
        if umax == 0:
            v[:] = 0
        else:
            u /= umax
            v = u.T.dot(aux) / u.T.dot(u)
            v = np.maximum(0, v)

        error_u.append(np.linalg.norm(u - u_old) / np.linalg.norm(u_old))
        error_v.append(np.linalg.norm(v - v_old) / np.linalg.norm(v_old))

        if error_u[-1] < tol and error_v[-1] < tol:
            break

    if ret_errors:
        return u, v, error_u, error_v
    else:
        return u, v


def nmu_admm(array, max_iter=5e2, tol=1e-3,
    init_strategy='svd', update_u=True, update_v=True,
    init_factors=None, ret_errors=False, eps=0):

    # allow user to specify initial factors and whether both are updated
    if init_factors is not None:
        u, v = init_factors
    else:
        u, v = _nmu_initialize(array, init_strategy=init_strategy)

    sigma = 1.
    gamma_r = np.zeros(array.shape)
    remainder = np.maximum(0, array - u.dot(v))

    # Alternating optimization
    error_u = []
    error_v = []
    error_rem = []
    for _ in range(int(max_iter)):
        u_old = u.copy()
        v_old = v.copy()
        # updating u, v:
        aux = array - remainder + gamma_r / sigma

        if update_u:
            u = aux.dot(v.T)
            u = np.maximum(0, u)
            umax = u.max()
            if umax <= 1e-10:
                u[:] = 0
                v[:] = 0
                break
            u /= umax

        if update_v:
            v = u.T.dot(aux) / u.T.dot(u)
            v = np.maximum(0, v)

        temp = array - u.dot(v)
        remainder = (sigma * temp + gamma_r) / (1 + sigma)
        remainder = np.maximum(eps, remainder)
        gamma_r += sigma * (temp - remainder)

        error_u.append(np.linalg.norm(u - u_old) / np.linalg.norm(u_old))
        error_v.append(np.linalg.norm(v - v_old) / np.linalg.norm(v_old))
        if ret_errors:
            error_rem.append(np.linalg.norm(array - remainder)
                             / np.linalg.norm(array))

        if error_u[-1] < tol and error_v[-1] < tol:
            break

    if ret_errors:
        return u, v, error_u, error_v, error_rem
    else:
        return u, v

def nmu_admm_with_baseline(array, max_iter=5e2, tol=1e-3,
    stim_start=100, stim_end=200, init_strategy='svd',
    update_u=True, update_v=True, update_mu=True,
    init_factors=None, ret_errors=False):

    # allow user to specify initial factors and whether both are updated
    if init_factors is not None:
        u, v, mu = init_factors
        assert v[:,0:stim_start] == 0.0, "Require opsin component to be zero before stim."
        
    else:
        u, v = _nmu_initialize(array, init_strategy=init_strategy)

        # zero out v before the stim starts
        v[:,0:stim_start] = 0.0
        mu = np.mean(array[:,0:stim_start], axis=-1, keepdims=True)

    sigma = 1.
    gamma_r = np.zeros(array.shape)
    remainder = np.maximum(0, array - u.dot(v) - mu)

    # Alternating optimization
    error_u = []
    error_v = []
    error_mu = []
    error_rem = []
    for _ in range(int(max_iter)):
        u_old = u.copy()
        v_old = v.copy()
        mu_old = mu.copy()

        # updating u, v:
        aux = array - remainder + gamma_r / sigma

        if update_u:
            u = aux.dot(v.T)
            u = np.maximum(0, u)
            umax = u.max()
            if umax <= 1e-10:
                u[:] = 0
                v[:] = 0
                break
            u /= umax

        # Ensure that v[0:stim_start] = 0
        if update_v:
            v[:,stim_start:] = u.T.dot(aux[:,stim_start:]) / u.T.dot(u)
            v = np.maximum(0, v)

        # Update mu using pre-stim + stim periods
        Q = remainder - u.dot(v)
        mu = (sigma * np.sum(Q, axis=-1, keepdims=True) + \
            np.sum(gamma_r, axis=-1, keepdims=True)) / sigma
        mu = np.maximum(mu, 0)

        temp = array - u.dot(v) - mu
        remainder = (sigma * temp + gamma_r) / (1 + sigma)
        remainder = np.maximum(0, remainder)
        gamma_r += sigma * (temp - remainder)

        error_u.append(np.linalg.norm(u - u_old) / np.linalg.norm(u_old))
        error_v.append(np.linalg.norm(v - v_old) / np.linalg.norm(v_old))
        error_mu.append(np.linalg.norm(mu - mu_old) / np.linalg.norm(mu_old))
        if ret_errors:
            error_rem.append(np.linalg.norm(array - remainder)
                             / np.linalg.norm(array))

        if error_u[-1] < tol and error_v[-1] < tol:
            break

    if ret_errors:
        return u, v, mu, error_u, error_v, error_mu, error_rem
    else:
        return u, v, mu


def _nmu_initialize(array, init_strategy):
    if isinstance(init_strategy, np.ndarray):
        x = init_strategy.copy()
        y = x.T.dot(array) / np.dot(x.T, x)
        m = np.max(x)
        if m > 0:
            x /= m
        y *= m
    elif init_strategy == 'max':
        idx = np.argmax(np.sum(array, axis=0))
        x = array[:, idx][:, np.newaxis]
        y = x.T.dot(array) / np.dot(x.T, x)
    elif init_strategy == 'svd':
        x, s, y = sp_linalg.svds(array, 1)
        y *= s[0]
        if np.all(x <= 1e-10) and np.all(y <= 1e-10):
            x *= -1
            y *= -1
    else:
        raise ValueError('Unknown initialization method')

    m = np.max(x)
    if m > 0:
        x /= m
    y *= m

    return x, y