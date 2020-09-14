import matplotlib.pyplot as plt
from numpy import log10
import numpy as np
from numpy.random import multinomial

from scaling.power_law_functions import pareto_occurencies_to_zipf

T = 100000  # time steps
N = 1000  # subjects
alpha = 100  # unit of wealth

wealth = multinomial(alpha, [1 / N] * N)  # first assignment random
for t in range(1, T):
    if t % 1000 == 0:
        print(f"iteration {t}")
    # random assignment of wealth among population
    # wealth += multinomial(alpha, [1 / N] * N)
    # assignment with hard preferential attachment
    # wealth += multinomial(alpha, wealth / wealth.sum())
    # assignment with partial preferential attachment
    wealth += multinomial(
        alpha,
        (wealth / wealth.sum() * (alpha * t) + np.array([1 / N] * N) * alpha)
        / (alpha * (t + 1)),
    )

plt.figure(1)
plt.title("wealth fraction distribution")
count, bins, ignored = plt.hist(wealth / wealth.sum(), bins=100)
# count, bins, ignored = plt.hist( log10(wealth[wealth > 0] / wealth.sum())) #, bins=100)
# plt.xscale('log')
# plt.yscale('log')

plt.figure(2)
plt.title("wealth fraction distribution log-log")
# count, bins, ignored = plt.hist( wealth / wealth.sum(), bins=100)
count, bins, ignored = plt.hist(log10(wealth[wealth > 0] / wealth.sum()))  # , bins=100)
# plt.xscale('log')
plt.yscale("log")

log_rank, log_s = pareto_occurencies_to_zipf()

plt.figure(3)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s)  # , 'yo', log_rank, poly1d_fn(log_rank), '--k')

plt.show()
