import matplotlib.pyplot as plt
from numpy import log10
import numpy as np

from scaling.power_law_functions import fit_pareto_alpha

a, m = 2., 1.0  # shape and mode
s = np.random.pareto(a, 1000000) + m  # x ~ x ^ - (a + 1)

# pareto
count, bins, ignored = plt.hist(s, 1000, normed=True)
fit = a * m ** a / bins ** (a + 1)

plt.figure(1)
plt.title(f"Pareto dstr linear a: {a}")
# plt.plot(bins, max(count) * fit / max(fit), linewidth=2, color='r')
plt.xscale("log")
plt.yscale("log")

plt.figure(2)
plt.title(f"Pareto dstr log-log a: {a}")
plt.hist(log10(s), 1000, normed=True)
# plt.xscale('log')
plt.yscale("log")
# plt.grid()

alpha_hat, alpha_hat_err = fit_pareto_alpha(s, x_min=None, return_error=True)
print(f"fitted alpha exponent: {alpha_hat} ± {alpha_hat_err}")


# zipf
log_s = -np.sort(-log10(s))
rank = np.array(range(len(log_s))) + 1
log_rank = log10(rank)

coef = np.polyfit(log_rank, log_s, 1)
poly1d_fn = np.poly1d(coef)

plt.figure(3)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s, "yo", log_rank, poly1d_fn(log_rank), "--k")
plt.grid()

plt.show()
