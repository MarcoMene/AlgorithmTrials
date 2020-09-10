import matplotlib.pyplot as plt
from numpy import log10, log, exp, sqrt, fabs
import numpy as np
from numpy.random import exponential, multinomial, normal, geometric, poisson, lognormal
from scaling.power_law import fit_pareto_alpha

# parameters

waiting_times = []

for n in range(1000):
    record = 0
    # t_record = 0
    for t in range(1, 100000 + 1):
        x = fabs(normal())
        if x > record:
            record = x
            if t > 1:
                waiting_times.append(t - t_record)
            t_record = t

# # ML fit to alpha exponent
# w_min = 10**1.6
alpha_hat, alpha_hat_err = fit_pareto_alpha(waiting_times, x_min=100, return_error=True)
print(f"fitted alpha exponent: {alpha_hat} ± {alpha_hat_err}")


plt.figure(0)
plt.title("waiting_times distribution")
count, bins, ignored = plt.hist(waiting_times, bins=1000)  #
plt.xscale('log')
plt.yscale('log')
plt.xlabel("waiting_times")
plt.ylabel("count")
plt.grid()

plt.figure(1)
plt.title("waiting_times distribution")
count, bins, ignored = plt.hist(log10(waiting_times)) #, bins=100)  #
# plt.xscale('log')
plt.yscale('log')
plt.xlabel("log waiting_times")
plt.ylabel("log count")
plt.grid()

plt.show()
