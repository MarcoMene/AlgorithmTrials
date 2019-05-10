from numpy.random import binomial
import matplotlib.pyplot as plt

N = 100
p = 0.99
size = 100000

samples = binomial(N, p, size) / N


plt.hist(samples, 50, normed=1, facecolor='g', alpha=0.75)

plt.xlabel('samples')
plt.ylabel('dP/dX')
plt.title('custom distribution samples')
plt.title(f"N: {N}, p: {p}, [size {size}]")
print("Enjoy the plot!")
plt.show()