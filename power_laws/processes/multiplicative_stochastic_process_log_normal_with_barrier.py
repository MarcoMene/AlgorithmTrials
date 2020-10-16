import matplotlib.pyplot as plt
from numpy import log10, exp
import numpy as np

from distrubutions.gamma_distribution import compute_params_gamma
from power_laws.power_law_functions import fit_pareto_alpha, pareto_occurencies_to_zipf

T = 5000  # time steps
N = 5000  # trajectories

sigma_1 = 0.1  # multiplicative noise
mu = 0.0  # drift

w0 = 0.08  # barrier
print(f"theoretical temperature T: {1 - w0}")
print(f"theoretical alpha exponent: {1 + 1 / (1 - w0)}")

Sts = []
time_of_last_lower_bound_hit = np.array([0] * N)
for n in range(N):
    St = 1.0
    for t in range(1, T + 1):
        dW1 = sigma_1 * np.random.normal()
        St += St * (mu + dW1)

        w_mean = exp(mu * t)
        w_th = w_mean * w0

        if St < w_th:
            St = w_th
            time_of_last_lower_bound_hit[n] = t

    Sts.append(St)

time_since_last_lower_bound_hit = T - time_of_last_lower_bound_hit

S_cutoff = w0 * exp(mu * T)
print(f"lower cutoff  w0 * w_mean: { S_cutoff }, in log10 {log10(S_cutoff)}")

Sts = np.array(Sts)

mask = Sts > S_cutoff
Sts = Sts[mask]
time_since_last_lower_bound_hit = time_since_last_lower_bound_hit[mask]

alpha_hat, alpha_hat_err = fit_pareto_alpha(Sts, x_min=10 ** -0.6, return_error=True)
print(f"fitted alpha exponent: {alpha_hat} ± {alpha_hat_err}")

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

# # transform to rank
log_rank, log_s = pareto_occurencies_to_zipf(Sts)

plt.figure(3)
plt.plot(log_rank, log_s)
plt.xlabel("log rank")
plt.ylabel("log St")
plt.grid()

plt.figure(4)
plt.title("time_since_last_lower_bound_hit distribution")
count, bins, ignored = plt.hist(time_since_last_lower_bound_hit / T, bins=50)  #
plt.yscale("log")
plt.xlabel("time_since_last_lower_bound_hit")
plt.ylabel("log count")

plt.show()
