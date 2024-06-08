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

######## PCA
from sklearn.decomposition import PCA

prcomp = PCA().set_output(transform='pandas')
PC_data = prcomp.fit_transform(milkscaled)
PC_data = PC_data.iloc[:,:2]
PC_data['Clust'] = km.labels_
PC_data['Clust'] = PC_data['Clust'].astype(str)

sns.scatterplot(data=PC_data, x='pca0', y='pca1',
                hue='Clust')
for i in np.arange(0, milk.shape[0] ):
    plt.text(PC_data.values[i,0], 
             PC_data.values[i,1], 
             list(milk.index)[i])
plt.show()