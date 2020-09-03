import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps


if __name__ == "__main__":
    s = np.random.lognormal(3., 0.5, 1000)

    count, bins, ignored = plt.hist(s, 50, normed=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
