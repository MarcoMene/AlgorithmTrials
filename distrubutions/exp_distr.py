import numpy as np
import matplotlib.pyplot as plt
from numpy import log10, log, power
from numpy.random import exponential



xs = exponential(2, size=100000)

plt.figure(1)
count, bins, ignored = plt.hist( xs, bins=100)
plt.yscale('log')
plt.xlabel("x")



plt.show()
