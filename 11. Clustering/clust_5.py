import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score 
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV 

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']


scaler = StandardScaler().set_output(transform='pandas')
X_scaled=scaler.fit_transform(X)


clusters = [2,3,4,5]
score = []
for c in clusters:
    clust = KMeans(random_state=23, n_clusters=c)
    clust.fit(X_scaled)
    sc = silhouette_score(X_scaled, clust.labels_)
    score.append(sc)
    
pd_score = pd.DataFrame({'Number':clusters,
                         'Score':score})
pd_score.sort_values('Score', ascending=False)

#### best k
km = KMeans(random_state=23, n_clusters=2)
km.fit(X_scaled)

X_clust = X.copy()
X_clust['Clust'] = km.labels_
print(X_clust['Clust'].value_counts())
X_clust[X_clust['Clust']==1]

#### GCV on whole data

kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
x_gbm = XGBClassifier(random_state=23)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth':[1,3, 5, None],
          'n_estimators': [50, 100, 150]}
gcv_x_gbm = GridSearchCV(x_gbm, param_grid=params,
                      verbose=3,cv=kfold, scoring='neg_log_loss')
gcv_x_gbm.fit(X, y)
print(gcv_x_gbm.best_params_)
print(gcv_x_gbm.best_score_)

##### GCV w/o outlier

X_wo = X[X_clust['Clust']==0]
y_wo = y[X_clust['Clust']==0]
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
x_gbm = XGBClassifier(random_state=23)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth':[1,3, 5, None],
          'n_estimators': [50, 100, 150]}
gcv_x_gbm = GridSearchCV(x_gbm, param_grid=params,
                      verbose=3,cv=kfold, scoring='neg_log_loss')
gcv_x_gbm.fit(X_wo, y_wo)
print(gcv_x_gbm.best_params_)
print(gcv_x_gbm.best_score_)


################# Glass ############################
glass = pd.read_csv("Glass.csv")
X = glass.drop('Type', axis=1)
y = glass['Type']

scaler = StandardScaler().set_output(transform='pandas')
X_scaled=scaler.fit_transform(X)

clusters = [2,3,4,5]
score = []
for c in clusters:
    clust = KMeans(random_state=23, n_clusters=c)
    clust.fit(X_scaled)
    sc = silhouette_score(X_scaled, clust.labels_)
    score.append(sc)
    
pd_score = pd.DataFrame({'Number':clusters,
                         'Score':score})
pd_score.sort_values('Score', ascending=False)

#### best k
km = KMeans(random_state=23, n_clusters=2)
km.fit(X_scaled)

X_clust = X.copy()
X_clust['Clust'] = km.labels_
print(X_clust['Clust'].value_counts())
X_clust[X_clust['Clust']==1]

pd.crosstab(index=y, columns=X_clust['Clust'])









