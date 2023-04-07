import matplotlib.pyplot as plt
from numpy import log, exp, sqrt, log10, pi
import numpy as np

T = 1000  # time steps
N = 10000  # subjects

mu = 0.0
sigma = 0.1

S0 = 1

S0s = np.array([S0] * N)

s_mean_on_realizations = np.array([0.] * N)

Ss = S0s
for t in range(T):
    s = np.random.normal(mu - sigma * sigma / 2, sigma, N)
    Ss = Ss * exp(s)

    s_mean_on_realizations += s

print(f"s_mean_on a realization on T {s_mean_on_realizations[0] / T * T}")


print("across realizations statistics")
print(f"S expected mean {S0 * exp(mu * T)}, median {S0 * exp((mu - sigma * sigma / 2) * T) }")
print(f"S actual mean {Ss.mean()}, median {np.median(Ss)}")

print(f"log S expected mean and median {(mu - sigma * sigma / 2) * T}, sigma {sigma * sqrt(T)}")
print(f"log S actual mean {log(Ss / S0s).mean()}, median {np.median(log(Ss / S0s))}, sigma {np.std(log(Ss / S0s))}")

plt.figure(1)
plt.title("S fraction distribution")
count, bins, ignored = plt.hist(Ss, bins=100)
plt.axvline(x=Ss.mean(), color='r')
plt.xlabel("S")
plt.ylabel("count")

plt.figure(2)
plt.title("S fraction distribution log - log")
plt.hist(Ss, bins=np.logspace(np.log10(Ss.min() * 0.5), np.log10(Ss.max() * 1.5), 100))
plt.axvline(x=Ss.mean(), color='r')
plt.xscale('log')  # , basex=exp(1))
plt.yscale("log")  # , basey=exp(1))
plt.xlabel("S [log]")
plt.ylabel("count [log]")

plt.figure(3)
plt.title("hist of logarithm of weath")
plt.hist(log(Ss), 100)
plt.axvline(x=log(Ss / S0s).mean(), color='r')
# plt.xscale('log')#, basex=exp(1))
plt.yscale("log")  # , basey=exp(1))
plt.xlabel("log S")
plt.ylabel("count [log]")

plt.figure(4)
plt.title("mean log returns * T on all N realizations")
plt.hist(s_mean_on_realizations, 100)
plt.axvline(x=(mu - sigma * sigma / 2) * T, color='r', label="expected (mu - sigma^2 / 2) * T")
plt.yscale("log")
plt.xlabel("mean log(S/S0)*T")
plt.ylabel("count [log]")


plt.show()
