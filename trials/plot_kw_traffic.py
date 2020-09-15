import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("/Users/marcomeneghelli/PycharmProjects/AlgorithmTrials/trials/data/keywords_traffic_sept20_gb.csv")

s = data.traffic


plt.figure(2)
plt.title(f"Keyword's impressions (log binning)")
count, bins, ignored = plt.hist(s, 96, normed=True, label="histogram of traffic")
plt.yscale("log")
plt.xlabel("traffic")

plt.show()
