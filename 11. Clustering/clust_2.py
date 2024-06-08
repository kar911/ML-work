import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score 

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
milkscaled=scaler.fit_transform(milk)

clust = AgglomerativeClustering(n_clusters=4, 
                                linkage='average')
clust.fit(milkscaled)
print(clust.labels_)
print(silhouette_score(milkscaled, clust.labels_))

milk_clust = milk.copy()
milk_clust['Clust'] = clust.labels_
milk_clust.sort_values('Clust')

clusters = [2,3,4,5,6]
score = []
for c in clusters:
    clust = AgglomerativeClustering(n_clusters=c, 
                                    linkage='ward')
    clust.fit(milkscaled)
    sc = silhouette_score(milkscaled, clust.labels_)
    score.append(sc)
    
pd_score = pd.DataFrame({'Number':clusters,
                         'Score':score})
pd_score.sort_values('Score', ascending=False)


