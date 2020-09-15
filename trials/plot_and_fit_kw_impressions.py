import matplotlib.pyplot as plt
from numpy import log10, power
import numpy as np

from scaling.power_law_functions import fit_pareto_alpha, pareto_occurencies_to_zipf
import pandas as pd


def C_from_point(alpha, x0, y0):
    return y0 / power(10, -alpha * x0)


def power_law_semilog(x, alpha, C):
    return C * power(10, -alpha * x)


data = pd.read_csv("/Users/marcomeneghelli/PycharmProjects/AlgorithmTrials/trials/data/keywords_traffic_sept20_it.csv")

s = data.impressions

x_min_for_fit = 10
alpha_hat, alpha_hat_err = fit_pareto_alpha(s, x_min=x_min_for_fit, return_error=True)
print(f"fitted alpha exponent: {round(alpha_hat, 2)} ± {round(alpha_hat_err, 2)}")

plt.figure(2)
plt.title(f"Keyword's impressions (log binning)")
count, bins, ignored = plt.hist(log10(s), 96, normed=True, label="histogram of impressions")
plt.yscale("log")
plt.xlabel("log-impressions")
idx_x_min = np.abs(bins - log10(x_min_for_fit)).argmin()

C = C_from_point(alpha_hat - 1, bins[idx_x_min], count[idx_x_min])
plt.plot(bins[(idx_x_min):], power_law_semilog(bins[(idx_x_min):], alpha_hat - 1, C), "--k",
         label="power-law fit")
plt.axvline(x=log10(x_min_for_fit), linewidth=2, color='r', label="min value for fit")
plt.legend()

# zipf

log_rank, log_s = pareto_occurencies_to_zipf(s, x_min=x_min_for_fit)
# idx_s_min = np.abs(log_s - log10(x_min_for_fit)).argmin()

plt.figure(3)
plt.title(f"Keyword's log impressions vs log rank (impressions >= {x_min_for_fit})")
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s, "yo", log_rank,
         -1 / (alpha_hat - 1) * log_rank + log_s[-1] + 1 / (alpha_hat - 1) * log_rank[-1], "--k")
plt.grid()
plt.xlabel("log-rank")
plt.ylabel("log-impressions")

plt.show()
