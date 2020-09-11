import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import levy_stable
from numpy import abs, log10

fig, ax = plt.subplots(1, 1)

alpha, beta = 1.4, 0.05

x = np.linspace(0.001, 500, 100)
# ax.plot(x, 2*levy_stable.pdf(x, alpha, beta),
#        'r-', lw=5, alpha=0.6, label='levy_stable pdf')

r = levy_stable.rvs(alpha, beta, size=100000)


count, bins, ignored = plt.hist(
    log10(abs(r)), 100, normed=True, label="levy_stable sampling"
)
#
# plt.figure(1)
# plt.xscale('log')
plt.yscale("log")
plt.legend()

plt.show()
