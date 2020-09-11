import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf

s = np.random.normal(1, 1, 10000)
# s = np.random.laplace(0, 1, 10000)
# s = np.random.lognormal(1, 1, 10000)


# MA(1) to get a non-flat spectrum
# s0 = np.random.normal(1, 1, 10000)
# s = s0[:-1] + s0[1:]

plt.figure(1)
count, bins, ignored = plt.hist(s, 50, normed=True)
plt.yscale("log")

#  timeseries
plt.figure(2)
plt.plot(s)

# spectral density (power spectrum)
plt.figure(3)
f, Pxx_den = signal.periodogram(s)
plt.semilogy(f, Pxx_den)
# plt.ylim([1e-7, 1e2])
plt.xlabel("frequency")
plt.ylabel("PSD")

plot_acf(s)

plt.show()
