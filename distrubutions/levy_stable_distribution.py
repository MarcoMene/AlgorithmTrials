import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import levy_stable
from numpy import abs, log10

fig, ax = plt.subplots(1, 1)

alpha, beta = 1.4, 0.0

# r =
# r2 = levy_stable.rvs(alpha, beta, size=100000)
# r3 = levy_stable.rvs(alpha, beta, size=100000)


plt.hist(
    levy_stable.rvs(2, 0, size=100000),
    bins=np.linspace(-10, 10, 100),
    alpha=0.5,
    normed=False, label="levy_stable a = 2 (Gaussian)"
)
plt.hist(
    levy_stable.rvs(1.5, 0, size=100000),
    bins=np.linspace(-10, 10, 100),
    alpha=0.5,
    normed=False, label="levy_stable a = 1.5",
    color="green"
)
plt.hist(
    levy_stable.rvs(1, 0, size=100000),
    bins=np.linspace(-10, 10, 100),
    alpha=0.5,
    normed=False, label="levy_stable a = 1 (Cauchy)",
    color="red"
)
plt.hist(
    levy_stable.rvs(0.5, 0, size=100000),
    bins=np.linspace(-10, 10, 100),
    alpha=0.5,
    normed=False, label="levy_stable a = 0.5",
    color="yellow"
)
plt.yscale("log")
plt.legend()

plt.show()
