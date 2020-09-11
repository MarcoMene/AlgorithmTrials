import matplotlib.pyplot as plt
from numpy import log10, log, exp, sqrt
import numpy as np
from numpy.random import exponential, multinomial, normal
from scipy import signal
from distrubutions.gamma_distribution import compute_params_gamma

T_growth = 1000
T_death = 100
N = 100000  # subjects

# death_times = np.array([T_growth] * N) # uniform distributed time of growth
death_times = exponential(T_death, N)
# sizes_when_they_die = exp(death_times/T_growth)  # deterministic transform


# log-normal growth
sizes_when_they_die = exp(
    normal(0.85 * death_times - 0.5 * death_times, sqrt(death_times))
)

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
log_s = -np.sort(-log10(sizes_when_they_die))
rank = np.array(range(len(log_s))) + 1
log_rank = log10(rank)

plt.figure(2)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s)  # , 'yo', log_rank, poly1d_fn(log_rank), '--k')
plt.xlabel("log rank")
plt.ylabel("log size")

plt.show()
