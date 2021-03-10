import autograd.numpy as np
from scipy.optimize import minimize
from autograd import grad

epsilon = 1e-50

def calc_excition(timestamps, timestamp_dims, to_dim, omega, T):
    if to_dim is not None:
        to_timestamps = np.append(timestamps[timestamp_dims == to_dim], T)
    else:
        to_timestamps = np.append(timestamps, T)
    diff = np.tile(to_timestamps, (timestamps.shape[0], 1)) - timestamps[:, np.newaxis]
    mat_excition = np.exp(-omega * diff) * (diff > 0)
    return mat_excition


def hawkes_likelihood(alpha_u, mat_excition, timestamp_dims, omega, sign, rho):
    mat_excition_alpha = alpha_u[timestamp_dims]
    term1 = np.sum(np.log(np.sum(mat_excition[:, :-1] * mat_excition_alpha[:, np.newaxis], axis=0) + epsilon))
    term2 = np.sum(mat_excition_alpha * (1 - mat_excition[:, -1])) / omega
    # regularizer
    regularizer = rho * np.sum(alpha_u ** 2)
    return sign * (term1 - term2 - regularizer)

def optimize_exo(timestamp_dims, to_dim, dim, omega, edge, mat_excition, rho):
    # global funs, mat_excitions
    neighbor_dims = np.arange(dim)[edge[:, to_dim] == 1]
    neighbor_timestamp_idxs = np.isin(timestamp_dims, neighbor_dims)

    mat_excition = mat_excition[neighbor_timestamp_idxs]

    timestamp_dims = timestamp_dims[neighbor_timestamp_idxs]

    result_alpha = np.zeros(dim)
    if mat_excition.shape[0] == 0:
        return result_alpha, -np.inf
    bounds = [(0, None) for _ in range(neighbor_dims.size)]
    optimization_param_id = {}
    for neighbor_dim in neighbor_dims:
        optimization_param_id[neighbor_dim] = len(optimization_param_id)
    optimization_timestamp_dims = np.array([optimization_param_id[dim] for dim in timestamp_dims])

    alpha_u = np.random.uniform(0, 1, size=neighbor_dims.size)
    res = minimize(hawkes_likelihood, alpha_u, args=(mat_excition, optimization_timestamp_dims, omega, -1, rho, False),
				   method="L-BFGS-B",
				   # jac=hawkes_likelihood_grad_exo,
                   jac=grad(hawkes_likelihood),
				   bounds=bounds,
				   options={"ftol": 1e-10, "maxls": 50, "maxcor":50, "maxiter":100, "maxfun": 100})
    return res.x
