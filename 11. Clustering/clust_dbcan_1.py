import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN 
from sklearn.metrics import silhouette_score

milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
milkscaled=scaler.fit_transform(milk)

clust = DBSCAN(eps=0.8, min_samples=3)
clust.fit(milkscaled)

print(clust.labels_)

epsilons = np.linspace(0.5, 1, 20)
min_pts = [2,3,4,5]
scores = []
for e in epsilons:
    for m in min_pts:
        clust = DBSCAN(eps=e, min_samples=m)
        clust.fit(milkscaled)
        c_data = milkscaled.copy()
        if len(np.unique(clust.labels_)) > 2:
            c_data['label'] = clust.labels_
            inliers = c_data[c_data['label'] != -1]
            sil = silhouette_score(inliers.iloc[:,:-1], 
                                 inliers['label'])
            scores.append([e,m,sil])

pd_scores = pd.DataFrame(scores, columns=['eps','min','sil'])
print("Best Params & Score:")
pd_scores.sort_values(by='sil', ascending=False).iloc[0]

#############
clust = DBSCAN(eps=0.973684, min_samples=3)
clust.fit(milkscaled)
c_data = milkscaled.copy()
c_data['label'] = clust.labels_
print("Outliers:")
c_data[clust.labels_==-1].index

################## Nutrient #######################
nut = pd.read_csv("nutrient.csv", index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
nutscaled=scaler.fit_transform(nut)

epsilons = np.linspace(0.5, 1, 20)
min_pts = [2,3,4,5]
scores = []
for e in epsilons:
    for m in min_pts:
        clust = DBSCAN(eps=e, min_samples=m)
        clust.fit(nutscaled)
        c_data = nutscaled.copy()
        if len(np.unique(clust.labels_)) > 2:
            c_data['label'] = clust.labels_
            inliers = c_data[c_data['label'] != -1]
            sil = silhouette_score(inliers.iloc[:,:-1], 
                                 inliers['label'])
            scores.append([e,m,sil])

pd_scores = pd.DataFrame(scores, columns=['eps','min','sil'])
print("Best Params & Score:")
pd_scores.sort_values(by='sil', ascending=False).iloc[0]

#############
clust = DBSCAN(eps=0.5, min_samples=2)
clust.fit(nutscaled)
c_data = nutscaled.copy()
c_data['label'] = clust.labels_
print("Outliers:")
c_data[clust.labels_==-1].index
print("Inliers:")
c_data[clust.labels_!=-1].index
