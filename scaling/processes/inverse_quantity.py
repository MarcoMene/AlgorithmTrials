import matplotlib.pyplot as plt
from numpy import log10, log, exp
import numpy as np
from numpy.random import exponential, multinomial, normal


N = 100000  # subjects

y = normal(size=N)
x = 1 / (y * y * y)  # deterministic transform


plt.figure(1)
plt.title("sizes_when_they_die distribution")
# count, bins, ignored = plt.hist( sizes_when_they_die, bins=100)  #
count, bins, ignored = plt.hist(log10(x), bins=100)  #
# plt.xscale('log')
plt.yscale("log")
plt.xlabel("log size")
plt.ylabel("log count")

# # transform to rank
log_s = -np.sort(-log10(x))
rank = np.array(range(len(log_s))) + 1
log_rank = log10(rank)

plt.figure(2)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s)  # , 'yo', log_rank, poly1d_fn(log_rank), '--k')
plt.xlabel("log rank")
plt.ylabel("log size")


plt.show()
