import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from numpy import vectorize
from scipy.stats import norm
import scipy.misc

# get empirical moments
print [norm.moment(i) for i in range(10)]


def approx_moment_generating_function(t, n=4, distribution=norm):
    """
    :type distribution: rv_continuous
    """
    res = 0
    for k in range(n+1):
        res += t**k * distribution.moment(k)/scipy.misc.factorial(k)
    return res

ts = np.linspace(-10, 10, 600)

plt.ylim([0,10])
plt.plot(ts, approx_moment_generating_function(ts), label="order 4")
plt.plot(ts, approx_moment_generating_function(ts, n=8), color="red", label="order 8")
plt.plot(ts, approx_moment_generating_function(ts, n=15), color="pink", label="order 15")
plt.legend()
print "Enjoy the plot!"
plt.show()
