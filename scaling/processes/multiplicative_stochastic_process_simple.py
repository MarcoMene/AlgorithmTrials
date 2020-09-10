import matplotlib.pyplot as plt
from numpy import log10, log
import numpy as np
from numpy.random import multinomial
from scipy import signal

from distrubutions.gamma_distribution import compute_params_gamma

T = 5000  # time steps
N = 1000  # subjects


mean_add = 0.0
std_add = 0.01

wealth = np.array([1] * N)
for t in range(1, T):
    if t % 1000 == 0:
        print(f"iteration {t}")
    s = np.random.normal(mean_add, std_add, N)
    wealth = wealth * ( 1 + s )


plt.figure(1)
plt.title("wealth fraction distribution")
count, bins, ignored = plt.hist(wealth, bins=100)  #
# count, bins, ignored = plt.hist( log10(wealth[wealth > 0] / wealth.sum())) #, bins=100)
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel("wealth")
plt.ylabel("count")

plt.figure(2)
plt.title("wealth fraction distribution log - log")
# count, bins, ignored = plt.hist( wealth/ wealth.sum() , bins=100)  # / wealth.sum()
# count, bins, ignored = plt.hist(wealth)  # , bins=100)
count, bins, ignored = plt.hist( log10(wealth )) #, bins=100)
# plt.xscale('log')
plt.yscale('log')
plt.xlabel("log wealth")
plt.ylabel("log count")

plt.show()
