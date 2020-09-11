import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.misc
from scipy.fftpack import fft, ifft, rfft, irfft
from fourier import (
    get_human_representation_of_function,
    format_function_for_fft,
    get_positive_xs_domain,
    get_human_representation_of_xs,
)


# get empirical moments
# print [norm.moment(i) for i in range(10)]


def approx_moment_generating_function(t, n=4, distribution=norm):
    """
    :type distribution: rv_continuous
    """
    res = 0
    for k in range(n + 1):
        res += t ** k * distribution.moment(k) / scipy.misc.factorial(k)
    return res


def approx_characteristic_function(t, n=4, distribution=norm):
    """
    :type distribution: rv_continuous
    """
    res = 0
    for k in range(n + 1):
        res += ((1j * t) ** k) * distribution.moment(k) / scipy.misc.factorial(k)
    return res


if __name__ == "__main__":
    # ts = np.linspace(-10, 10, 600)
    #
    # plt.ylim([0, 10])
    # plt.plot(ts, approx_moment_generating_function(ts), label="order 4")
    # plt.plot(ts, approx_moment_generating_function(ts, n=8), color="red", label="order 8")
    # plt.plot(ts, approx_moment_generating_function(ts, n=15), color="pink", label="order 15")
    # plt.legend()
    # print "Enjoy the plot!"
    # plt.show()

    fs = np.linspace(0, 2.3, 30)
    fts = format_function_for_fft(approx_characteristic_function, fs, n=16)

    fys = fft(fts, n=3000)
    iys = ifft(fts, n=3000)
    # iys = irfft(np.real(fts))

    plt.figure(1)
    plt.subplot(311)
    plt.title("Fourier transform")

    h_fxs, h_fys = get_human_representation_of_function(
        get_positive_xs_domain(fys), fys
    )

    plt.plot(np.real(h_fys), color="blue", label="Re")
    plt.plot(np.imag(h_fys), color="red", label="Im")
    plt.plot(
        np.sqrt(np.real(h_fys) ** 2 + np.imag(h_fys) ** 2),
        color="green",
        label="module",
    )
    plt.legend()

    plt.subplot(312)
    plt.title("Characteristic function")
    h_fs, h_ys = get_human_representation_of_function(fs, fts)

    plt.plot(h_fs, np.real(h_ys), color="blue", label="Re")
    plt.plot(h_fs, np.imag(h_ys), color="red", label="Im")
    plt.plot(
        h_fs,
        np.sqrt(np.real(h_ys) ** 2 + np.imag(h_ys) ** 2),
        color="green",
        label="module",
    )
    plt.legend()

    plt.subplot(313)
    plt.title("Rebuilt pdf")

    h_ixs, h_iys = get_human_representation_of_function(
        get_positive_xs_domain(iys), iys
    )

    plt.plot(np.real(h_iys), color="blue", label="Re")
    plt.plot(np.imag(h_iys), color="red", label="Im")
    plt.plot(
        np.sqrt(np.real(h_iys) ** 2 + np.imag(h_iys) ** 2),
        color="green",
        label="module",
    )

    # h_xs = get_human_representation_of_xs(fs)
    # plt.plot(norm.pdf(h_xs), color="black",ls='dashed', label="true pdf")

    plt.legend()

    plt.show()
