import numpy as np
import matplotlib.pyplot as plt
from numpy import log10, log


if __name__ == "__main__":
    s = np.random.lognormal(3.0, 0.5, 1000)

    count, bins, ignored = plt.hist(log10(s), 50, normed=True)
    # plt.xscale('log')
    plt.yscale("log")
    plt.show()
