import matplotlib.pyplot as plt
from numpy import log10
import numpy as np

from scaling.power_law import fit_pareto_alpha

alpha = 2.2

a, m = alpha - 1, 1.0  # shape and mode # x ~ x ^ - (a + 1)

Ns = [10, 100, 200, 500, 1000, 2000, 5000, 10000]
n = 30

means = {N: [] for N in Ns}
for N in Ns:
    for i in range(N):
        s = np.random.pareto(a, N) + m
        means[N].append(s.mean())

print(f"theoretical mean {a*m/(a-1)}")

plt.figure(1)
plt.title(f"Histogram of mean, N {Ns[-1]}")
plt.hist(means[Ns[-1]], 100, normed=True)
plt.xscale("log")
plt.yscale("log")
plt.grid()

plt.figure(2)
plt.plot(Ns, [np.array(means[N]).mean() for N in Ns])
plt.xlabel("sample size")
plt.ylabel("mean of the means")

plt.show()
