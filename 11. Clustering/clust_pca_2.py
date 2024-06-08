import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 

usa = pd.read_csv("USArrests.csv",index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
usascaled=scaler.fit_transform(usa)

clusters = [2,3,4,5,6]
score = []
for c in clusters:
    clust = KMeans(random_state=23, n_clusters=c)
    clust.fit(usascaled)
    sc = silhouette_score(usascaled, clust.labels_)
    score.append(sc)
    
pd_score = pd.DataFrame({'Number':clusters,
                         'Score':score})
pd_score.sort_values('Score', ascending=False)

#### best k
km = KMeans(random_state=23, n_clusters=2)
km.fit(usascaled)

usa_clust = usa.copy()
usa_clust['Clust'] = km.labels_
print(usa_clust.groupby('Clust').mean())

######## PCA
from sklearn.decomposition import PCA

prcomp = PCA().set_output(transform='pandas')
PC_data = prcomp.fit_transform(usascaled)
PC_data = PC_data.iloc[:,:2]
PC_data['Clust'] = km.labels_
PC_data['Clust'] = PC_data['Clust'].astype(str)

sns.scatterplot(data=PC_data, x='pca0', y='pca1',
                hue='Clust')
for i in np.arange(0, usa.shape[0] ):
    plt.text(PC_data.values[i,0], 
             PC_data.values[i,1], 
             list(usa.index)[i])
plt.show()
