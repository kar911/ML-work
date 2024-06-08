import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans 

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
milkscaled=scaler.fit_transform(milk)

clusts = np.arange(2,25)
wss = []
for c in clusts:
    km = KMeans(random_state=23, n_clusters=c)
    km.fit(milkscaled)
    wss.append(km.inertia_)

plt.scatter(clusts, wss, c='red')
plt.plot(clusts, wss)
plt.xlabel("No. of Clusters")
plt.ylabel("WSS")
plt.show()

################# USArrests ###################

usa = pd.read_csv("USArrests.csv",index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
usascaled=scaler.fit_transform(usa)


clusts = np.arange(2,25)
wss = []
for c in clusts:
    km = KMeans(random_state=23, n_clusters=c)
    km.fit(usascaled)
    wss.append(km.inertia_)

plt.scatter(clusts, wss, c='red')
plt.plot(clusts, wss)
plt.xlabel("No. of Clusters")
plt.ylabel("WSS")
plt.show()