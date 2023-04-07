import matplotlib.pyplot as plt
from numpy import log, exp, sqrt, log10, pi, float_power, fabs
import numpy as np
from scipy.stats import levy_stable, linregress
from scipy.special import gamma


T = 1000  # time steps
N = 1000  # subjects

mu = 0.01
sigma = 0.1

alpha = 2.

S0 = 1

S0s = np.array([S0] * N)

ts_to_probe = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, T]
variances = []
S_at_zero = []

s_t = np.array([0.] * N)


def return_to_origin_calculator(a, c, n):
    return gamma(1 / a) / (pi * a * float_power(fabs(c * n), 1 / a))


Ss = S0s
for t in range(1, T + 1):
    s = levy_stable.rvs(alpha, 0, loc=mu - float_power(sigma, alpha) / 2, scale=sigma / sqrt(2), size=N)
    # s = np.random.normal(mu - float_power(sigma, alpha) / 2, sigma, N)

    s_t += s

    # Ss = Ss * exp(s)
    if t in ts_to_probe:
        print(f"iteration {t}")
        var = s_t.var()
        variances.append(var)

        # TODO: fir levy stable
        # use fitted a, c

        # S_at_zero.append(1 / sqrt(2 * pi * var))
        S_at_zero.append(return_to_origin_calculator(alpha, sigma / sqrt(2), t))

Ss = S0s * exp(s_t)

print(f"S expected mean {S0 * exp(mu * T)}, median {S0 * exp((mu - float_power(sigma, alpha) / 2) * T) }")
print(f"S actual mean {Ss.mean()}, median {np.median(Ss)}")

print(f"log S expected mean and median {(mu - float_power(sigma, alpha) / 2) * T}, sigma {sigma * float_power(T, 1 / alpha)}")
print(f"log S actual mean {s_t.mean()}, median {np.median(s_t)}")

# plt.figure(1)
# plt.title("S fraction distribution")
# count, bins, ignored = plt.hist(Ss, bins=100)
# plt.xlabel("S")
# plt.yscale("log")
# plt.ylabel("count")
#
# plt.figure(2)
# plt.title("S fraction distribution log - log")
# plt.hist(Ss,
#          bins=np.logspace(np.log10(Ss.min() * 0.5), np.log10(Ss.max() * 1.5), 100))
# plt.xscale('log')
# plt.yscale("log")
# plt.xlabel("S [log]")
# plt.ylabel("count [log]")

plt.figure(3)
plt.title("hist of logarithm of S")
plt.hist(s_t, 100)
plt.yscale("log")
plt.xlabel("log S")
plt.ylabel("count [log]")

plt.figure(4)
plt.title("Scaling of variance")
plt.plot(ts_to_probe, variances, marker="o")
print(linregress(ts_to_probe, variances / float_power(sigma, 2)))
plt.xlabel("t")
plt.ylabel("var( log(S / S0)  )")

plt.figure(5)
plt.title("Scaling of density at zero")
plt.plot(ts_to_probe, S_at_zero, marker="o")
print(linregress(log(ts_to_probe), log(S_at_zero)))
plt.xscale("log")
plt.yscale("log")
plt.xlabel("t")
plt.ylabel("density at zero: f_S(0)")

plt.show()
