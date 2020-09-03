import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf
import colorednoise as cn

beta = 1  # the exponent  1: pink
# beta = 0.5  # the exponent  2: brown
s = cn.powerlaw_psd_gaussian(beta, 1000000)

plt.figure(1)
count, bins, ignored = plt.hist(s, 50, normed=True)
plt.yscale('log')

plt.figure(2)
plt.plot(s)

plt.figure(3)
f, Pxx_den = signal.periodogram(s)
# plt.semilogy(f, Pxx_den)
plt.loglog(f, Pxx_den)
# plt.ylim([1e-7, 1e2])
plt.xlabel('frequency')
plt.ylabel('PSD')

plot_acf(s)

plt.show()
