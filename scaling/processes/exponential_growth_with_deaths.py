import matplotlib.pyplot as plt
from numpy import log10, log, exp
import numpy as np
from numpy.random import exponential, multinomial
from scipy import signal


T_growth = 100
T_death = 100
N = 10000 # subjects

death_times = exponential(T_death, N)
sizes_when_they_die = exp(death_times/T_growth)


plt.figure(1)
plt.title("sizes_when_they_die distribution")
# count, bins, ignored = plt.hist( sizes_when_they_die, bins=100)  #
count, bins, ignored = plt.hist( log10(sizes_when_they_die), bins=100)  #
# plt.xscale('log')
plt.yscale('log')
plt.xlabel("log size")
plt.ylabel("log count")

# # transform to rank
log_s = -np.sort(-log10(sizes_when_they_die))
rank = np.array(range(len(log_s))) + 1
log_rank = log10(rank)

plt.figure(2)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s) #, 'yo', log_rank, poly1d_fn(log_rank), '--k')
plt.xlabel("log rank")
plt.ylabel("log size")


plt.show()
