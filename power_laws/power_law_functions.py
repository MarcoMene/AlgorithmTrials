import numpy as np
from numpy.core._multiarray_umath import log10, log


def pareto_occurencies_to_zipf(xs, x_min=None):
    if isinstance(xs, list):
        xs = np.array(xs)
    # ML fit to alpha exponent
    x_min = x_min or xs.min()
    xs_for_fit = xs[xs >= x_min]

    log_s = -np.sort(-log10(xs_for_fit))
    rank = np.array(range(len(log_s))) + 1
    log_rank = log10(rank)
    return log_rank, log_s


def fit_pareto_alpha(xs, x_min=None, return_error=False):
    if isinstance(xs, list):
        xs = np.array(xs)
    # ML fit to alpha exponent
    x_min = x_min or xs.min()
    xs_for_fit = xs[xs >= x_min]
    alpha_hat = 1 + xs_for_fit.size / log(xs_for_fit / x_min).sum()
    if return_error:
        alpha_hat_err = (alpha_hat - 1) / np.sqrt(xs_for_fit.size)
        return alpha_hat, alpha_hat_err
    return alpha_hat
