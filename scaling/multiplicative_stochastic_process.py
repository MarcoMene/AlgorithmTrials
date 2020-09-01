import matplotlib.pyplot as plt
from numpy import log10
import numpy as np
from numpy.random import multinomial

from distrubutions.gamma_distribution import _compute_params_gamma

T = 10000  # time steps
N = 10000  # subjects
alpha = 100000  # unit of wealth
alpha0 = 1  # minimum wealth

# mumtiplicative factor distribution
mean_l = 1.01
sigma_l = 0.02
shape, scale = _compute_params_gamma(mean_l, sigma_l)


wealth = multinomial(alpha, [1 / N] * N) + alpha0  # first assignment random
for t in range(1, T):
    if t % 1000 == 0:
        print(f"iteration {t}")
    s = np.random.gamma(shape, scale, N)
    wealth = wealth * s
    alpha = wealth.sum()/N
    wealth[wealth < alpha] = alpha

plt.figure(1)
plt.title("wealth fraction distribution")
count, bins, ignored = plt.hist( wealth/ wealth.sum(), bins=100)  #
# count, bins, ignored = plt.hist( log10(wealth[wealth > 0] / wealth.sum())) #, bins=100)
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel("wealth")
plt.ylabel("count")

plt.figure(2)
plt.title("wealth fraction distribution log-log")
# count, bins, ignored = plt.hist( wealth/ wealth.sum() , bins=100)  # / wealth.sum()
count, bins, ignored = plt.hist( log10(wealth / wealth.sum())) #, bins=100)
# plt.xscale('log')
plt.yscale('log')
plt.xlabel("log wealth")
plt.ylabel("log count")

# transform to rank
log_s = -np.sort(-log10(wealth/ wealth.sum()))
rank = np.array(range(len(log_s))) + 1
log_rank = log10(rank)
#
# coef = np.polyfit(log_rank, log_s, 1)
# poly1d_fn = np.poly1d(coef)

plt.figure(3)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s) #, 'yo', log_rank, poly1d_fn(log_rank), '--k')
plt.xlabel("log rank")
plt.ylabel("log wealth")

plt.show()
