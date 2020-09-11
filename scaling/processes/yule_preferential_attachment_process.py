import matplotlib.pyplot as plt
from numpy import log10, log, exp, sqrt
import numpy as np
from numpy.random import exponential, multinomial, normal, geometric, poisson, lognormal
from scaling.power_law import fit_pareto_alpha, pareto_occurencies_to_zipf

# parameters
k0 = 1  # initial wealth
c = 0  # initial bias in  wealth
m = 10  # new wealth per time-step

T = 100000

wealth = np.array([k0 + c])
for t in range(T):
    wealth = np.append(wealth, k0 + c)
    wealth += multinomial(poisson(m), wealth / wealth.sum())
    # wealth += multinomial(int(exponential(m)), wealth / wealth.sum())
    # wealth += multinomial(int(round(lognormal(m, sqrt(m)))), wealth / wealth.sum())  # <-- no Yule

    # uniform assignment
    # wealth += multinomial(int(exponential(m)), [1/len(wealth)] * len(wealth))


print(f"theoretical exponent alpha {2 + (k0 + c) / m}")

# ML fit to alpha exponent
w_min = 10 ** 1.6
alpha_hat, alpha_hat_err = fit_pareto_alpha(wealth, x_min=w_min, return_error=True)
print(f"fitted alpha exponent: {alpha_hat} ± {alpha_hat_err}")


plt.figure(0)
plt.title("wealth distribution")
count, bins, ignored = plt.hist(wealth, bins=1000)  #
plt.xscale("log")
plt.yscale("log")
plt.xlabel("wealth")
plt.ylabel("count")
plt.grid()


plt.figure(1)
plt.title("wealth distribution")
count, bins, ignored = plt.hist(log10(wealth), bins=100)  #
# plt.xscale('log')
plt.yscale("log")
plt.xlabel("log wealth")
plt.ylabel("log count")
plt.grid()

# # transform to rank
log_rank, log_s = pareto_occurencies_to_zipf(wealth)

plt.figure(2)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s)  # , 'yo', log_rank, poly1d_fn(log_rank), '--k')
plt.xlabel("log rank")
plt.ylabel("log wealth")
plt.grid()

plt.show()
