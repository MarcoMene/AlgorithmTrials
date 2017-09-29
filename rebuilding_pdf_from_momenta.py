import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from numpy import vectorize
from scipy.stats import norm
import scipy.misc
from scipy.fftpack import fft, ifft
from fourier import get_human_representation_of_function, format_function_for_fft, get_positive_xs_domain


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
        res += (1j * t) ** k * distribution.moment(k) / scipy.misc.factorial(k)
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

    xs = np.linspace(0, 10, 200)
    ys = format_function_for_fft(approx_characteristic_function, xs, n=8)

    fys = fft(ys)
    iys = ifft(ys)

    print fys
    print iys


    plt.figure(1)
    plt.subplot(311)
    plt.title('Fourier transform')

    h_fs, h_fys = get_human_representation_of_function(get_positive_xs_domain(fys), fys)

    plt.plot(h_fs, np.real(h_fys), color="blue", label="Re")
    plt.plot(h_fs, np.imag(h_fys), color="red", label="Im")
    plt.plot(h_fs, np.sqrt(np.real(h_fys) ** 2 + np.imag(h_fys) ** 2), color="green", label="module")
    plt.legend()

    plt.subplot(312)
    plt.title('Characteristic function')
    h_xs, h_ys = get_human_representation_of_function(xs, ys)

    plt.plot(h_xs, np.real(h_ys), color="blue", label="Re")
    plt.plot(h_xs, np.imag(h_ys), color="red", label="Im")
    plt.plot(h_xs, np.sqrt(np.real(h_ys) ** 2 + np.imag(h_ys) ** 2), color="green", label="module")
    plt.legend()

    plt.subplot(313)
    plt.title('Rebuilt pdf')

    h_iys, h_iys = get_human_representation_of_function(get_positive_xs_domain(iys), iys)

    plt.plot(h_iys, np.real(h_iys), color="blue", label="Re")
    plt.plot(h_iys, np.imag(h_iys), color="red", label="Im")
    plt.plot(h_iys, np.sqrt(np.real(h_iys) ** 2 + np.imag(h_iys) ** 2), color="green", label="module")
    plt.legend()

    print "Enjoy the plot!"
    plt.show() 
