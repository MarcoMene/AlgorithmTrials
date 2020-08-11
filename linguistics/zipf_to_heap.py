import matplotlib.pyplot as plt
import numpy as np
from numpy.random import zipf

a = 2.1

# ns = list(np.arange(10, 1000000, 100))  # [10, 100, 1000, 10000, 100000, 1000000]
ns = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
vs = [None] * len(ns)

text = zipf(a, size=ns[0])
vs[0] = len(np.unique(text))
for i in range(1, len(ns)):
    add_text = zipf(a, size=ns[i] - ns[i - 1])
    text = np.concatenate((text, add_text))
    vs[i] = len(np.unique(text))

plt.figure(2)
plt.scatter(ns, vs)

plt.show()
