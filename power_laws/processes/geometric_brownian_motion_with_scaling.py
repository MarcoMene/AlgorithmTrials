import matplotlib.pyplot as plt
from numpy import log, exp, sqrt, log10, pi
import numpy as np

T = 1000  # time steps
N = 1000  # subjects

mu = 0.01
sigma = 0.1

S0 = 1

S0s = np.array([S0] * N)

ts_to_probe = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, T-1]
variances = []
S_at_zero = []

Ss = S0s
for t in range(T):
    s = np.random.normal(mu - sigma * sigma / 2, sigma, N)
    Ss = Ss * exp(s)
    if t in ts_to_probe:
        print(f"iteration {t}")
        var = log(Ss / S0s).var()
        variances.append(var)
        S_at_zero.append(  1/sqrt(2*pi*var) )


print(f"S expected mean {S0 * exp(mu * T)}, median {S0 * exp((mu - sigma * sigma / 2) * T) }")
print(f"S actual mean {Ss.mean()}, median {np.median(Ss)}")

print(f"log S expected mean and median {(mu - sigma * sigma / 2) * T}, sigma {sigma * sqrt(T)}")
print(f"log S actual mean {log(Ss / S0s).mean()}, median {np.median(log(Ss / S0s))}")

# plt.figure(1)
# plt.title("S fraction distribution")
# count, bins, ignored = plt.hist(Ss, bins=100)  #
# plt.xlabel("S")
# plt.ylabel("count")
#
# plt.figure(2)
# plt.title("S fraction distribution log - log")
# count, bins, ignored = plt.hist(Ss,
#                                 bins=np.logspace(np.log10(Ss.min() * 0.5), np.log10(Ss.max() * 1.5), 100))
# plt.xscale('log')  # , basex=exp(1))
# plt.yscale("log")  # , basey=exp(1))
# plt.xlabel("S [log]")
# plt.ylabel("count [log]")
#
# plt.figure(3)
# plt.title("hist of logarithm of weath")
# count, bins, ignored = plt.hist(log(Ss), 100)
# # plt.xscale('log')#, basex=exp(1))
# plt.yscale("log")  # , basey=exp(1))
# plt.xlabel("log S")
# plt.ylabel("count [log]")


plt.figure(4)
plt.title("Scaling of variance")
plt.plot(ts_to_probe, variances, marker="o")
plt.xlabel("t")
plt.ylabel("var( log(S / S0)  )")


plt.figure(5)
plt.title("Scaling of density at zero")
plt.plot(ts_to_probe, S_at_zero, marker="o")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("t")
plt.ylabel("density at zero: f_S(0)")

plt.show()
