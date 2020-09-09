import matplotlib.pyplot as plt
from numpy import log10, log, exp
import numpy as np
from numpy.random import exponential, multinomial, normal, geometric


N = 100000 # subjects

exp_rate = 0.1

w = exponential(1 / exp_rate, N)

ps = 1 - exp(-w)   # p: probability of dying

k = geometric(ps)





plt.figure(1)
plt.title("k distribution")
count, bins, ignored = plt.hist(log10(k), bins=100)  #
# plt.xscale('log')
plt.yscale('log')
plt.xlabel("log size")
plt.ylabel("log count")

# # transform to rank
log_s = -np.sort(-log10(k))
rank = np.array(range(len(log_s))) + 1
log_rank = log10(rank)

plt.figure(2)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s) #, 'yo', log_rank, poly1d_fn(log_rank), '--k')
plt.xlabel("log rank")
plt.ylabel("log size")

plt.figure(3)
plt.title("probabilities of dying")
count, bins, ignored = plt.hist(ps, bins=100)  #
plt.xlabel("p")
plt.ylabel("count")



plt.show()
