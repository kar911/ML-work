import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
milkscaled=scaler.fit_transform(milk)

km = KMeans(random_state=23, n_clusters=3)
km.fit(milkscaled)
print(km.labels_)
print(silhouette_score(milkscaled, km.labels_))

clusters = [2,3,4,5,6]
score = []
for c in clusters:
    clust = KMeans(random_state=23, n_clusters=c)
    clust.fit(milkscaled)
    sc = silhouette_score(milkscaled, clust.labels_)
    score.append(sc)
    
pd_score = pd.DataFrame({'Number':clusters,
                         'Score':score})
pd_score.sort_values('Score', ascending=False)

#### best k
km = KMeans(random_state=23, n_clusters=3)
km.fit(milkscaled)

milk_clust = milk.copy()
milk_clust['Clust'] = km.labels_
print(milk_clust.groupby('Clust').mean())

############### Nutrient ##############
nut = pd.read_csv("nutrient.csv", index_col=0)
scaler = StandardScaler().set_output(transform='pandas')
nut_scaled=scaler.fit_transform(nut)


clusters = [2,3,4,5,6]
score = []
for c in clusters:
    clust = KMeans(random_state=23, n_clusters=c)
    clust.fit(nut_scaled)
    sc = silhouette_score(nut_scaled, clust.labels_)
    score.append(sc)
    
pd_score = pd.DataFrame({'Number':clusters,
                         'Score':score})
pd_score.sort_values('Score', ascending=False)

#### best k
km = KMeans(random_state=23, n_clusters=3)
km.fit(nut_scaled)

nut_clust = nut.copy()
nut_clust['Clust'] = km.labels_
print(nut_clust.groupby('Clust').mean())

