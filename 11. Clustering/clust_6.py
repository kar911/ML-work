import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 

rfm = pd.read_csv("rfm_data_customer.csv", index_col=0)
rfm.drop('most_recent_visit', axis=1, inplace=True)

scaler = StandardScaler().set_output(transform='pandas')
rfm_scaled=scaler.fit_transform(rfm)


clusters = [2,3,4,5]
score = []
for c in clusters:
    clust = KMeans(random_state=23, n_clusters=c)
    clust.fit(rfm_scaled)
    sc = silhouette_score(rfm_scaled, clust.labels_)
    score.append(sc)
    
pd_score = pd.DataFrame({'Number':clusters,
                         'Score':score})
pd_score.sort_values('Score', ascending=False)

#### best k
km = KMeans(random_state=23, n_clusters=3)
km.fit(rfm_scaled)

rfm_clust = rfm.copy()
rfm_clust['Clust'] = km.labels_
print(rfm_clust.groupby('Clust').mean())

