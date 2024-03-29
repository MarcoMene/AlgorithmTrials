import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sps
from bsp_data_science.statistics.distributions_parameters import compute_params_gamma, compute_params_lognormal


# def compute_params_gamma(x: float, sx: float):
#     """Computes the parameters of a gamma distribution with mean x and
#     standard deviation sx. The names used in the dictionary as output are the
#     ones used by the scipy.stats module.
#
#     :param x +/- sx: variable with standard deviation
#     :return: dictionary containing the parameters of the distribution
#     """
#     k = x ** 2 / sx ** 2
#     theta = sx ** 2 / x
#     return k, theta

if __name__ == "__main__":
    shape, scale = compute_params_gamma(0.015, 0.018)
    s = np.random.gamma(shape, scale, 100000)

    count, bins, ignored = plt.hist(s, 500, density=True)
    # y = bins ** (shape - 1) * (
    #     np.exp(-bins / scale) / (sps.gamma(shape) * scale ** shape)
    # )
    # plt.plot(bins, y, linewidth=2, color="r")
    plt.yscale('log')
    plt.show()
