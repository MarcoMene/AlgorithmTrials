import matplotlib.pyplot as plt
from numpy.random import pareto
from numpy import log, exp
import numpy as np
from numpy.random import zipf

a = 1.01

N = 100000

# generate zipf
s = zipf(a, size=N)


def freq_count(x):
    ii, y = np.unique(x, return_counts=True)
    return ii, y / y.sum()


# sort items with rank
symbols, ps = freq_count(s)

xs = -log(ps)
# get x


plt.figure(0)
plt.scatter(symbols, ps)
plt.title("Zipf distribution")
plt.xscale("log")
plt.yscale("log")
plt.ylim((1 / N / 3, 1))

plt.figure(1)
plt.title("p distribution--> giving Ni")
count, bins, ignored = plt.hist(ps, bins=500, normed=True)
plt.xscale("log")
plt.yscale("log")


plt.figure(2)
plt.title("negative entropy x distribution")
count, bins, ignored = plt.hist(xs, bins=50, normed=True)
# plt.xscale('log')
plt.yscale("log")

plt.figure(3)
plt.title("exp(-x) distribution")
count, bins, ignored = plt.hist(exp(-xs), bins=500, normed=True)
# plt.xscale('log')
plt.yscale("log")

plt.show()
