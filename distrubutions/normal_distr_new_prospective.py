import numpy as np
import matplotlib.pyplot as plt
from numpy import log10, log, power
from numpy.random import exponential, normal



xs = normal(size=10000)

# xs2 = power(xs, 2)

plt.figure(1)
count, bins, ignored = plt.hist( xs, bins=100)
plt.yscale('log')
plt.xlabel("x")

plt.figure(2)
plt.plot(power(bins[1:], 2), count )
plt.yscale('log')


plt.show()
