import matplotlib.pyplot as plt
from numpy import log10, power
import numpy as np

from scaling.power_law_functions import fit_pareto_alpha, pareto_occurencies_to_zipf
import pandas as pd


def C_from_point(alpha, x0, y0):
    return y0 / power(10, -alpha * x0)


def power_law_semilog(x, alpha, C):
    return C * power(10, -alpha * x)


data = pd.read_csv("/Users/marcomeneghelli/PycharmProjects/AlgorithmTrials/trials/data/keywords_traffic_sept20_gb.csv")

s = data.traffic


plt.figure(2)
plt.title(f"Keyword's impressions (log binning)")
count, bins, ignored = plt.hist(s, 96, normed=True, label="histogram of traffic")
plt.yscale("log")
plt.xlabel("traffic")

plt.show()
