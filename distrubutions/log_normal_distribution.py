import numpy as np
import matplotlib.pyplot as plt
from numpy import log10, log, exp


if __name__ == "__main__":
    mean_underlying_normal = 0.
    sigma_underlying_normal = 10

    size = 100000

    s = np.random.lognormal(mean_underlying_normal, sigma_underlying_normal, size)

    print(f"expected median {exp(mean_underlying_normal)} , actual count {len(s[s>= exp(mean_underlying_normal)])/size}")

    count, bins, ignored = plt.hist(log10(s), 1000, normed=True)
    # plt.xscale('log')
    plt.yscale("log")
    plt.show()
