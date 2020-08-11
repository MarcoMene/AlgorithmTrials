import matplotlib.pyplot as plt
from numpy import log10
import numpy as np

a, m = 2.1, 1.  # shape and mode
s = np.random.pareto(a, 10000) + m

# pareto
count, bins, ignored = plt.hist(s, 1000, normed=True)
fit = a * m ** a / bins ** (a + 1)

plt.figure(1)
plt.plot(bins, max(count) * fit / max(fit), linewidth=2, color='r')
plt.xscale('log')
plt.yscale('log')

# zipf
log_s = -np.sort(-log10(s))
rank = np.array(range(len(log_s))) + 1
log_rank = log10(rank)

coef = np.polyfit(log_rank, log_s, 1)
poly1d_fn = np.poly1d(coef)

plt.figure(2)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s, 'yo', log_rank, poly1d_fn(log_rank), '--k')

plt.show()