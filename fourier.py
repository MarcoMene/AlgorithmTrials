import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from numpy import vectorize


def gaussian(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) / (np.sqrt(2.0 * np.pi) * sigma)


def _finite_pulse(x, a=1):
    if -a <= x <= a:
        return 1
    return 0


def format_function_for_fft(function, xs=np.linspace(0, 10, 100), **pars):
    xs = np.concatenate([xs, -xs[::-1][:-1]])
    ys = function(xs, **pars)
    return ys


def get_human_representation_of_xs(xs):
    """
    accepts fft formatted data as ys
    original xs as xs
    """
    return np.concatenate([-xs[::-1][:-1], xs])

def get_human_representation_of_function(xs, ys):
    """
    accepts fft formatted data as ys
    original xs as xs
    """
    xs = get_human_representation_of_xs(xs)
    n = len(ys)
    ys = np.concatenate([ys[(n // 2 + 1):], ys[:(n // 2 + 1)]])
    return xs, ys


def get_positive_xs_domain(fys):
    return np.array(list(range(len(fys) // 2 + 1)))


finite_pulse = vectorize(_finite_pulse)

if __name__ == "__main__":
    xs = np.linspace(0, 10, 200)
    ys = format_function_for_fft(gaussian, xs, mu=0, sigma=.05)

    fys = fft(ys)
    iys = ifft(fys)

    print fys
    print iys

    h_fs, h_fys = get_human_representation_of_function(get_positive_xs_domain(fys), fys)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(h_fs, np.real(h_fys), color="blue", label="Re")
    plt.plot(h_fs, np.imag(h_fys), color="red", label="Im")
    plt.plot(h_fs, np.sqrt(np.real(h_fys) ** 2 + np.imag(h_fys) ** 2), color="green", label="module")
    plt.legend()

    plt.subplot(212)
    h_xs, h_ys = get_human_representation_of_function(xs, ys)

    # plt.plot(ys, color="black")
    plt.plot(h_xs, h_ys, color="black")
    # plt.plot(xs, np.real(ys), color="black")
    # plt.plot(xs, np.real(iys), color="pink")
    print "Enjoy the plot!"
    plt.show()
