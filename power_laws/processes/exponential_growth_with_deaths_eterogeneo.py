import matplotlib.pyplot as plt
from numpy import log10, log, exp, sqrt
import numpy as np
from numpy.random import exponential, multinomial, normal
from scipy import signal
from power_laws.power_law_functions import fit_pareto_alpha, pareto_occurencies_to_zipf
from scipy.stats import lognorm, gamma

T_g = 200    # mean time of growth
T_d = 100    # mean time of death
N = 100000  # subjects

# death_times = np.array([T_growth] * N) # uniform distributed time of growth
death_times = exponential(T_d, N)

# deterministic transform
# sizes_when_they_die = exp(death_times / T_g)


# distributed growth rate transform
mu_g = T_g
sigma_g = 10
Tgs = normal(loc=mu_g, scale=sigma_g, size=N)

# def compute_params_lognormal(x: float, sx: float) -> dict:
#     """ Computes the parameters of a lognormal distribution with mean x and
#     standard deviation sx. The names used in the dictionary as output are the
#     ones used by the scipy.stats module.
#
#     :param x +/- sx: variable with standard deviation
#     :return: dictionary containing the parameters of the distribution
#     """
#     mu = 2 * np.log(x) - 1 / 2 * np.log(x ** 2 + sx ** 2)
#     sigma = np.sqrt(np.log(x ** 2 + sx ** 2) - 2 * np.log(x))
#     return {"scale": np.exp(mu), "s": sigma}
#
# params = compute_params_lognormal(mu_g, sigma_g)
# Tgs = lognorm.rvs(size=N, **params)


# def compute_params_gamma(x: float, sx: float) -> dict:
#     """ Computes the parameters of a gamma distribution with mean x and
#     standard deviation sx. The names used in the dictionary as output are the
#     ones used by the scipy.stats module.
#
#     :param x +/- sx: variable with standard deviation
#     :return: dictionary containing the parameters of the distribution
#     """
#     k = x ** 2 / sx ** 2
#     theta = sx ** 2 / x
#     return {"scale": theta, "a": k}
#
# params = compute_params_gamma(mu_g, sigma_g)
# Tgs = gamma.rvs(size=N, **params)


sizes_when_they_die = exp(death_times / Tgs)
sizes_when_they_die = sizes_when_they_die[(sizes_when_they_die > 0) & (sizes_when_they_die < np.inf)]

# # ML fit to alpha exponent
alpha_hat, alpha_hat_err = fit_pareto_alpha(sizes_when_they_die, x_min=5, return_error=True)
print(f"expected (trivial) alpha exponent: {1 + mu_g/T_d}")
print(f"fitted alpha exponent: {alpha_hat} ± {alpha_hat_err}")


plt.figure(1)
plt.title("sizes_when_they_die distribution")
# count, bins, ignored = plt.hist( sizes_when_they_die, bins=100)  #
count, bins, ignored = plt.hist(log10(sizes_when_they_die), bins=100)  #
# count, bins, ignored = plt.hist(log10(sizes_when_they_die[(sizes_when_they_die > 0) & (sizes_when_they_die < np.inf)]), bins=100)  #
# plt.xscale('log')
plt.yscale("log")
plt.xlabel("log size")
plt.ylabel("log count")

# # transform to rank
# log_rank, log_s = pareto_occurencies_to_zipf(sizes_when_they_die)
#
# plt.figure(2)
# # plt.scatter(log_rank, log_s)
# plt.plot(log_rank, log_s)  # , 'yo', log_rank, poly1d_fn(log_rank), '--k')
# plt.xlabel("log rank")
# plt.ylabel("log size")

plt.show()
