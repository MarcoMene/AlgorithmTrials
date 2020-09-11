from scipy.stats import rv_continuous
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def gaussian_ratio_cdf(t, mu1=0.0, mu2=0.0, s1=1.0, s2=1.0, correlation=0.0):
    """
    This is valid only for ratio of positive quantities.

    Documentation:
    file:///Users/marcomeneghelli/Downloads/v16i04.pdf
    """
    cov_term = 0.0
    if correlation != 0.0:
        cov_term = 2 * t * correlation * s1 * s2
    return norm.cdf(
        (t * mu2 - mu1) / np.sqrt(s1 ** 2 - cov_term + (t ** 2) * (s2 ** 2))
    )


class GaussianRatioDistribution(rv_continuous):
    def _cdf(self, x, mu1, mu2, s1, s2):
        return gaussian_ratio_cdf(x, mu1, mu2, s1, s2)


def get_ratio_confidence_interval(mu1, mu2, s1, s2, confidence=0.95):
    my_dist = GaussianRatioDistribution()

    bounds = my_dist.ppf(
        np.array([1 - confidence, confidence]), mu1=mu1, mu2=mu2, s1=s1, s2=s2
    )
    return tuple(bounds)


xs = np.linspace(-10, 10, 200)

mu1 = 1.2
s1 = 0.1
mu2 = 1.0
s2 = 0.15

ci = get_ratio_confidence_interval(mu1, mu2, s1, s2, confidence=0.68)
ci2 = get_ratio_confidence_interval(mu1, mu2, s1, s2, confidence=0.95)
print ci
print ci2

my_dist = GaussianRatioDistribution()

# ys = np.linspace(0, 1, 100)
# inv_cdf = my_dist.ppf(ys, mu1=mu1, mu2=mu2, s1=s1, s2=s2)
# plt.plot(ys, inv_cdf)
# plt.show()

fs = my_dist.pdf(xs, mu1=mu1, mu2=mu2, s1=s1, s2=s2)
ifs = my_dist.cdf(xs, mu1=mu1, mu2=mu2, s1=s1, s2=s2)

med = my_dist.median(mu1=mu1, mu2=mu2, s1=s1, s2=s2)
print med
# ci = my_dist.interval(alpha=0.68, mu1=mu1, mu2=mu2, s1=s1, s2=s2)
# print ci
# ci2 = my_dist.interval(alpha=0.95, mu1=mu1, mu2=mu2, s1=s1, s2=s2)
# print ci2
plt.plot(xs, fs)
plt.plot(xs, ifs, color="red")
plt.axvline(x=med, color="blue")
plt.axvline(x=ci[0], color="green")
plt.axvline(x=ci[1], color="green")
plt.axvline(x=ci2[0], color="red")
plt.axvline(x=ci2[1], color="red")

# samples = my_dist.rvs(mu=mu, sigma=sigma, size=1000)
#
# plt.hist(samples, 50, normed=1, facecolor='g', alpha=0.75)
#
# plt.xlabel('samples')
# plt.ylabel('dP/dX')
# plt.title('custom distribution samples')
# plt.grid(True)
print "Enjoy the plot!"
plt.show()
