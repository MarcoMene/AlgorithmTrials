import matplotlib.pyplot as plt
from numpy import log, exp, sqrt, log10
import numpy as np
from scipy.stats import gamma

from scaling.power_law_functions import fit_pareto_alpha

T = 100  # time steps
N = 10000  # subjects

mu = 0.01
sigma = 0.1

S0 = 1


# def compute_params_gamma(x: float, sx: float) -> dict:
#     """ Computes the parameters of a gamma distribution with mean x and
#     standard deviation sx. The names used in the dictionary as output are the
#     ones used by the scipy.stats module.
#
#     :param x +/- sx: variable with standard deviation
#     :return: dictionary containing the parameters of the distribution
#     """
#     k = x ** 2 / sx ** 2
#     theta = sx ** 2 / x
#     return {"scale": theta, "a": k}
#
# gamma_params = compute_params_gamma(mu, sigma)


def slope_loglog_linear_intercept():
    return ((mu - sigma * sigma / 2)) / (sigma * sigma) - 1


def slope_loglog_quadratic_term(logx):
    return -logx / (sigma * sigma * T)


def slope_loglog(logx):
    return slope_loglog_linear_intercept() + slope_loglog_quadratic_term(logx)


wealth0 = np.array([S0] * N)

wealth = wealth0
for t in range(T):
    if t % 1000 == 0:
        print(f"iteration {t}")
    s = np.random.normal(mu - sigma * sigma / 2, sigma, N)
    # s = gamma.rvs(size=N, **gamma_params)
    wealth = wealth * exp(s)

print(f"wealth expected mean {S0 * exp(mu * T)}, median {S0 * exp((mu - sigma * sigma / 2) * T) }")
print(f"wealth actual mean {wealth.mean()}, median {np.median(wealth)}")

# wealth2 = wealth0 * exp(np.random.normal( (mu - sigma * sigma / 2)* T, sigma * sqrt(T), N))
# print(f"wealth2 direct attempt actual mean {wealth2.mean()}, median {np.median(wealth2)}")

print(f"log wealth expected mean and median {(mu - sigma * sigma / 2) * T}, sigma {sigma * sqrt(T)}")
print(f"log wealth actual mean {log(wealth / wealth0).mean()}, median {np.median(log(wealth / wealth0))}")
# print(f"log wealth2 actual mean {log(wealth2 / wealth0).mean()}, median {np.median(log(wealth2 / wealth0))}")


log_1sigma_right = (mu - sigma * sigma / 2) * T + sigma * sqrt(T)
alpha_hat = fit_pareto_alpha(wealth, x_min=S0 * exp(log_1sigma_right), return_error=False)
print(f"Approx alpha exponent: {alpha_hat}")

points_logx = [(mu - sigma * sigma / 2) * T + sigma * sqrt(T) * i for i in [0, 1, 2, 3]]
print(f"Slope at some points { points_logx } ")
print(f"Slope  {[(round(slope_loglog(logx), 2), round(slope_loglog_linear_intercept(), 2), round(slope_loglog_quadratic_term(logx), 2)) for logx in points_logx]} ")

plt.figure(1)
plt.title("wealth fraction distribution")
count, bins, ignored = plt.hist(wealth, bins=100)  #
plt.xlabel("wealth")
plt.ylabel("count")

plt.figure(2)
plt.title("wealth fraction distribution log - log")
count, bins, ignored = plt.hist(wealth,
                                bins=np.logspace(np.log10(wealth.min() * 0.5), np.log10(wealth.max() * 1.5), 100))
plt.xscale('log')#, basex=exp(1))
plt.yscale("log")  # , basey=exp(1))
plt.xlabel("wealth [log]")
plt.ylabel("count [log]")


plt.figure(3)
plt.title("hist of logarithm of weath")
count, bins, ignored = plt.hist(log(wealth), 100)
# plt.xscale('log')#, basex=exp(1))
plt.yscale("log")  # , basey=exp(1))
plt.xlabel("log wealth")
plt.ylabel("count [log]")


plt.show()
