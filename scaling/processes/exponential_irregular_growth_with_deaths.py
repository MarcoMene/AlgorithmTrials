import matplotlib.pyplot as plt
from numpy import log10, log, exp, sqrt
import numpy as np
from numpy.random import exponential, multinomial, normal, lognormal
from scipy import signal
from distrubutions.gamma_distribution import compute_params_gamma

# mumtiplicative factor distribution
mean_l = 1.01
sigma_l = 0.01
shape, scale = compute_params_gamma(mean_l, sigma_l)

T_growth = 100
T_death = 1
N = 100000  # subjects

death_times = exponential(T_death, N)
# sizes_when_they_die = exp( lognormal(mean=death_times/T_growth, sigma=0.1*sqrt(death_times/T_growth)  )   )  # scale is extracted from an exp distr
# sizes_when_they_die = exp( lognormal(mean=death_times/T_growth, sigma=0.1*sqrt(death_times/T_growth)  )   )  # scale is extracted from an exp distr
sizes_when_they_die = exp(
    exponential(death_times / T_growth)
)  # scale is extracted from an exp distr

plt.figure(1)
plt.title("sizes_when_they_die distribution")
# count, bins, ignored = plt.hist( sizes_when_they_die, bins=100)  #
count, bins, ignored = plt.hist(log10(sizes_when_they_die), bins=100)  #
# plt.xscale('log')
plt.yscale("log")
plt.xlabel("log size")
plt.ylabel("log count")

# # transform to rank
log_s = -np.sort(-log10(sizes_when_they_die))
rank = np.array(range(len(log_s))) + 1
log_rank = log10(rank)

plt.figure(2)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s)  # , 'yo', log_rank, poly1d_fn(log_rank), '--k')
plt.xlabel("log rank")
plt.ylabel("log size")


plt.show()
