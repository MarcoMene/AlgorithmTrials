import matplotlib.pyplot as plt
from numpy import log10, log
import numpy as np
from numpy.random import multinomial
from scipy import signal


from distrubutions.gamma_distribution import _compute_params_gamma

T = 50000  # time steps
N = 1000  # subjects
w_th = 100000  # unit of wealth
w0 = 0.285  # minimum wealth

print(f"theoretical temperature T: {1-w0}")
print(f"theoretical alpha exponent: {-1-1/(1-w0)}")

# mumtiplicative factor distribution
mean_l = 1.01
sigma_l = 0.01
shape, scale = _compute_params_gamma(mean_l, sigma_l)


wealth = np.array([1]*N) # multinomial(alpha, [1 / N] * N) + alpha0  # first assignment random
timeseries_log_returns = []
for t in range(1, T):
    if t % 1000 == 0:
        print(f"iteration {t}")
    s = np.random.gamma(shape, scale, N)
    # s = np.random.normal(mean_l, sigma_l, N)
    wealth = wealth * s
    # timeseries_log_returns.append( log(s).mean() )
    timeseries_log_returns.append( s[-1] )
    wealth = wealth * N / wealth.sum()

    w_mean = wealth.sum() / N
    w_th = w_mean * w0
    wealth[wealth < w_th] = w_th
    wealth = wealth * N / wealth.sum()


w_cutoff = w0 * wealth.sum() / N
print(f"lower cutoff  w0 * w_mean: { w_cutoff }")

# ML fit to alpha exponent
w_min = wealth.min()
# w_min = w_cutoff

w_for_fit = wealth[wealth>= w_min]
alpha_hat = 1 + w_for_fit.size / log(w_for_fit/w_min).sum()
print(f"fitted alpha exponent: {-alpha_hat}")



plt.figure(1)
plt.title("wealth fraction distribution")
count, bins, ignored = plt.hist( wealth, bins=100)  #
# count, bins, ignored = plt.hist( log10(wealth[wealth > 0] / wealth.sum())) #, bins=100)
# plt.xscale('log')
# plt.yscale('log')
plt.xlabel("wealth")
plt.ylabel("count")

plt.figure(2)
plt.title("wealth fraction distribution log-log")
# count, bins, ignored = plt.hist( wealth/ wealth.sum() , bins=100)  # / wealth.sum()
count, bins, ignored = plt.hist( log10(wealth )) #, bins=100)
# plt.xscale('log')
plt.yscale('log')
plt.xlabel("log wealth")
plt.ylabel("log count")

# transform to rank
log_s = -np.sort(-log10(wealth))
rank = np.array(range(len(log_s))) + 1
log_rank = log10(rank)
#
# coef = np.polyfit(log_rank, log_s, 1)
# poly1d_fn = np.poly1d(coef)

plt.figure(3)
# plt.scatter(log_rank, log_s)
plt.plot(log_rank, log_s) #, 'yo', log_rank, poly1d_fn(log_rank), '--k')
plt.xlabel("log rank")
plt.ylabel("log wealth")

plt.figure(4)
plt.plot(timeseries_log_returns)
plt.xlabel("time step")
plt.ylabel("mean wealth")

# plt.figure(5)
# f, Pxx_den = signal.periodogram(timeseries_log_returns)
# plt.loglog(f, Pxx_den)
# plt.xlabel('frequency')
# plt.ylabel('PSD')


plt.show()
