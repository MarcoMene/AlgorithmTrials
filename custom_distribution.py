from scipy.stats import rv_continuous
import numpy as np
import matplotlib.pyplot as plt

class MyCustomDistribution(rv_continuous):
    def _pdf(self, x, mu, sigma):
            return np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) / (np.sqrt(2.0 * np.pi) * sigma)


my_dist = MyCustomDistribution()

xs = np.linspace(-10, 10, 200)
fs = my_dist.pdf(xs, mu=0.0000000000000000001, sigma=3)
#
# print gaussian.median()
# print gaussian.interval(0.95)

plt.plot(xs, fs)
# plt.show()
#

samples = my_dist.rvs(mu=0.0000000000000000001, sigma=3, size=1000)

plt.hist(samples, 50, normed=1, facecolor='g', alpha=0.75)

plt.xlabel('samples')
plt.ylabel('dP/dX')
plt.title('custom distribution samples')
plt.grid(True)
plt.show()