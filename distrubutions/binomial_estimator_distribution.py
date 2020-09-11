from numpy.random import binomial
import matplotlib.pyplot as plt

N = 100000
size = 100000

p1 = 0.01
p2 = 1.0

samples1 = binomial(N, p1, size) / N
samples2 = binomial(N, p2, size) / N

ratios = samples1 / samples2

plt.hist(ratios, 50, normed=1, facecolor="g", alpha=0.75)

plt.xlabel("samples")
plt.ylabel("dP/dX")
plt.title("custom distribution samples")
plt.title(f"N: {N}, p1: {p1}, p2: {p2}, [size {size}]")
print("Enjoy the plot!")
plt.show()
