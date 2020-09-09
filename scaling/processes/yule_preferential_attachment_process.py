import matplotlib.pyplot as plt
from numpy import log10, log, exp, sqrt
import numpy as np
from numpy.random import exponential, multinomial, normal, geometric

# parameters
k0 = 1  # initial wealth
c = 0  # initial bias in  wealth
m = 10  # new wealth per time-step

T = 100000

wealth = np.array([k0 + c])
for t in range(T):
    wealth = np.append(wealth, k0 + c)
    wealth += multinomial(m, wealth/wealth.sum())

print(f"theoretical exponent alpha {2 + (k0 + c)/m}")

# ML fit to alpha exponent  # TODO: build a function
w_min = 10

w_for_fit = wealth[wealth>= w_min]
alpha_hat = 1 + w_for_fit.size / log(w_for_fit/w_min).sum()
print(f"fitted alpha exponent: {alpha_hat} ± {(alpha_hat - 1)/sqrt(w_for_fit.size)}")



plt.figure(1)
plt.title("wealth distribution")
count, bins, ignored = plt.hist(log10(wealth), bins=100)  #
# plt.xscale('log')
plt.yscale('log')
plt.xlabel("log wealth")
plt.ylabel("log count")

# # transform to rank
log_s = -np.sort(-log10(wealth))
rank = np.array(range(len(log_s))) + 1
log_rank = log10(rank)

plt.figure(2)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s)  # , 'yo', log_rank, poly1d_fn(log_rank), '--k')
plt.xlabel("log rank")
plt.ylabel("log wealth")

plt.show()
