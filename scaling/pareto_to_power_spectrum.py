import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pareto

fig, ax = plt.subplots(1, 1)

b = 0.3

# mean, var, skew, kurt = pareto.stats(b, moments='mvsk')
# print( pareto.stats(b, moments='mvsk'))

t_min, t_max = pareto.ppf(0.01, b), pareto.ppf(0.99, b)

ts = np.linspace(t_min, t_max, 100)

plt.figure(1)
plt.loglog(ts, pareto.pdf(ts, b), "r-", lw=5, alpha=0.6, label="pareto pdf")


def S(f, N=500):
    assert f > 0

    dt = 1 / f / N
    res = 0
    for i in range(N):
        res += dt * (i * dt) * pareto.pdf(i * dt, b)
    return res


fs = np.linspace(0.001, 1, 100)

plt.figure(2)
plt.loglog(
    fs, [S(f) for f in fs], "r-", lw=5, alpha=0.6, label="S(f) as rebuilt from S.O.C."
)

plt.show()
