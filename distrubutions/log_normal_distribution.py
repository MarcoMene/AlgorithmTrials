import numpy as np
import matplotlib.pyplot as plt
from numpy import log10, log, exp
from bsp_data_science.statistics.distributions_parameters import compute_params_gamma, compute_params_lognormal
import scipy


if __name__ == "__main__":
    # mean_underlying_normal = 0.
    # sigma_underlying_normal = 10
    #
    # size = 100000
    #
    # s = np.random.lognormal(mean_underlying_normal, sigma_underlying_normal, size)
    #
    # print(f"expected median {exp(mean_underlying_normal)} , actual count {len(s[s>= exp(mean_underlying_normal)])/size}")
    #
    # count, bins, ignored = plt.hist(log10(s), 1000, density=True)
    # # plt.xscale('log')
    # plt.yscale("log")
    # plt.show()

    pars = compute_params_lognormal(0.015, 0.018)

    xs = scipy.stats.lognorm.rvs(**pars, size=10000)

    print(f"sample mean, std {np.mean(xs), np.std(xs)}")

    count, bins, ignored = plt.hist(xs, 500, density=True)
    plt.yscale("log")
    plt.show()
