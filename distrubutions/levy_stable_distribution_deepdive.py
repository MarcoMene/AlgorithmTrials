import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import levy_stable
from numpy import abs, log10

fig, ax = plt.subplots(1, 1)

alpha, beta = 2, 0.0

mu = 0.5
sigma = 1

N = 100000

plt.hist(
    levy_stable.rvs(2, 0, loc= mu - np.float_power(sigma, alpha) / 2, scale=sigma / np.sqrt(2), size=N),
    bins=np.linspace(-5, 5, 100),
    alpha=0.5,
    normed=False,
    label=f"levy_stable a = 2 (Gaussian), loc, scale = {mu, sigma}"
)
plt.hist(
    np.random.normal(mu - np.float_power(sigma, alpha) / 2, sigma, N),
    bins=np.linspace(-5, 5, 100),
    alpha=0.5,
    normed=False,
    label=f"Regular Gaussian {mu, sigma}",
    color="red"
)
plt.yscale("log")
plt.legend()

plt.show()
