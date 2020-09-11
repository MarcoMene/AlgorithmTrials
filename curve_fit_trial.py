from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def exponential_plus_constant(x, a, b, c):
    return a * np.exp(-b * x) + c


# xdata = np.linspace(0, 4, 50)
xdata = np.linspace(0, 80, 80)
y = exponential_plus_constant(xdata, 0.16, 1 / 3, 0.01)
y_noise = 0.2 * y * np.random.normal(size=y.size)
ydata = y + y_noise

plt.plot(xdata, ydata, "b-", label="data")

popt, pcov = curve_fit(exponential_plus_constant, xdata, ydata)
plt.plot(xdata, exponential_plus_constant(xdata, *popt), "r-", label="fit")

# Constrain the optimization to the region of 0 < a < 3, 0 < b < 2 and 0 < c < 1:

popt, pcov = curve_fit(
    exponential_plus_constant,
    xdata,
    ydata,
    bounds=(0, [1.0, 100.0, 1.0]),
    absolute_sigma=False,
    sigma=(0.2 * y),
)
plt.plot(
    xdata,
    exponential_plus_constant(xdata, *popt),
    "g--",
    label="fit-with-bounds-and-sigmas-absolute_sigma=False",
)

popt2, pcov2 = curve_fit(
    exponential_plus_constant,
    xdata,
    ydata,
    bounds=(0, [1.0, 100.0, 1.0]),
    absolute_sigma=True,
    sigma=(0.2 * y),
)
plt.plot(
    xdata,
    exponential_plus_constant(xdata, *popt2),
    "y--",
    label="fit-with-bounds-and-sigmas-absolute_sigma=True",
)

print("Foreseen value @ {}:".format(81))
print(exponential_plus_constant(81, *popt))

print("Foreseen value @ {}:".format(8100))
print(exponential_plus_constant(8100, *popt))

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
print("Enjoy the plot!")
plt.show()
