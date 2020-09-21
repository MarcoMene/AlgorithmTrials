import matplotlib.pyplot as plt
from numpy import log10, log, exp, sqrt
import numpy as np
from numpy.random import exponential, multinomial, normal
from scipy import signal
from distrubutions.gamma_distribution import compute_params_gamma
from scaling.power_law_functions import fit_pareto_alpha, pareto_occurencies_to_zipf

T_g = 200    # mean time of growth
T_d = 100    # mean time of death
N = 100000  # subjects

# death_times = np.array([T_growth] * N) # uniform distributed time of growth
death_times = exponential(T_d, N)
sizes_when_they_die = exp(death_times / T_g)  # deterministic transform


# log-normal growth
# mu = 0.85
# sigma = 1
# sizes_when_they_die = exp(
#     normal(mu * death_times - 0.5 * sigma * sigma * death_times, sigma * sqrt(death_times))
# )

# # ML fit to alpha exponent
alpha_hat, alpha_hat_err = fit_pareto_alpha(sizes_when_they_die, x_min=10, return_error=True)
print(f"fitted alpha exponent: {alpha_hat} ± {alpha_hat_err}")


# probabilistic transform
# sizes_when_they_die = []
# mean_l = 1.2
# sigma_l = 0.01
# shape, scale = compute_params_gamma(mean_l, sigma_l)
# for td in death_times:
#     factors = np.random.gamma(shape, scale, int(round(td)))
#     f = factors.cumprod()[-1] if len(factors) > 1 else 1
#     sizes_when_they_die.append(f)


plt.figure(1)
plt.title("sizes_when_they_die distribution")
# count, bins, ignored = plt.hist( sizes_when_they_die, bins=100)  #
count, bins, ignored = plt.hist(log10(sizes_when_they_die), bins=100)  #
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
