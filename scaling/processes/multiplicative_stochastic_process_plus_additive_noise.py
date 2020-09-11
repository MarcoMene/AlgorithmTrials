import matplotlib.pyplot as plt
from numpy import log10
import numpy as np

from distrubutions.gamma_distribution import compute_params_gamma
from scaling.power_law import fit_pareto_alpha

T = 100  # time steps
N = 1000  # trajectories

sigma_1 = 0.01  # multiplicative noise
sigma_2 = 0.01  # additive noise
mu = 0.01  # drift


Sts = []
for n in range(N):
    St = 1.0
    for t in range(1, T):
        dW1 = sigma_1 * np.random.normal()
        dW2 = sigma_2 * np.random.normal()
        St += St * (mu + dW1) + dW2
    Sts.append(St)


# alpha_hat = fit_pareto_alpha()
# print(f"fitted alpha exponent: {-alpha_hat}")

plt.figure(1)
plt.title("St distribution")
count, bins, ignored = plt.hist(Sts, bins=100)  #
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel("St")
plt.ylabel("count")

plt.figure(2)
plt.title("St distribution log-log")
count, bins, ignored = plt.hist(log10(Sts), bins=100)
plt.yscale("log")
plt.xlabel("log St")
plt.ylabel("log count")

plt.show()
