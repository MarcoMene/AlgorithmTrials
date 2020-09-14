import numpy as np
import matplotlib.pyplot as plt
from numpy import log10, log


if __name__ == "__main__":
    s = np.random.lognormal(0., 10, 100000)

    count, bins, ignored = plt.hist(log10(s), 1000, normed=True)
    # plt.xscale('log')
    plt.yscale("log")
    plt.show()
