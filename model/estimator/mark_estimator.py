import autograd.numpy as np
from scipy.optimize import minimize
from autograd import grad

epsilon = 1e-50

def calc_excition(timestamps, timestamp_dims, to_dim, mark, v):
    if to_dim is not None:
        to_timestamps = timestamps[timestamp_dims == to_dim]
    else:
        to_timestamps = timestamps
    diff = np.tile(to_timestamps, (timestamps.shape[0], 1)) - timestamps[:, np.newaxis]
    mat_excition = mark[:, np.newaxis] * np.exp(-v * diff) * (diff > 0)
    return mat_excition


def mark_likelihood(beta_u, mat_excition, mark, sentiments, timestamp_dims, sign, rho):
    mat_excition_beta = beta_u[timestamp_dims]
    intensity = np.sum(mat_excition_beta[:, np.newaxis] * mat_excition, axis=0)
    term1 = mark * intensity
    term2 = np.log(np.sum(np.exp(np.tile(sentiments[:, np.newaxis], (1, intensity.size)) * intensity), axis=0))
    # regularizer
    regularizer = rho * np.sum(beta_u ** 2)
    return sign * (np.sum(term1 - term2) - regularizer)


def optimize(timestamp_dims, mark, to_dim, dim, edge, mat_excition, rho, sentiments):
    neighbor_dims = np.arange(dim)[edge[:, to_dim] == 1]
    neighbor_timestamp_idxs = np.isin(timestamp_dims, neighbor_dims)

    mat_excition = mat_excition[neighbor_timestamp_idxs]

    timestamp_dims = timestamp_dims[neighbor_timestamp_idxs]

    if mat_excition.shape[0] == 0:
        return np.zeros(dim), -np.inf
    bounds = [(1e-15,None) for _ in range(neighbor_dims.size)]
    optimization_param_id = {}
    result_beta = np.zeros(dim)
    if mat_excition.shape[0] == 0:
        return result_beta, -np.inf
    for neighbor_dim in neighbor_dims:
        optimization_param_id[neighbor_dim] = len(optimization_param_id)
    optimization_timestamp_dims = np.array([optimization_param_id[dim] for dim in timestamp_dims])

    beta_u = np.random.uniform(0, 1, size=neighbor_dims.size)
    res = minimize(mark_likelihood, beta_u, args=(mat_excition, mark, sentiments, optimization_timestamp_dims, -1.0, rho),
                   method="L-BFGS-B",
                   jac=grad(mark_likelihood),
                   bounds=bounds,
                   options={"ftol": 1e-10, "maxls": 50, "maxcor":50, "maxiter":100, "maxfun": 100})
    result_beta[neighbor_dims] = res.x
    result_beta[result_beta < 0] = 0
    return result_beta, -res.fun
