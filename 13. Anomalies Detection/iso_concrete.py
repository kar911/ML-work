import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

concrete = pd.read_csv("Concrete_Data.csv")


clf = IsolationForest(contamination=0.05,
                      random_state=23)
clf.fit(concrete)
predictions = clf.predict(concrete)

print("%age of outliers="+ str((predictions<0).mean()*100)+ "%")
abn_ind = np.where(predictions < 0)

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 

inliers = concrete[predictions != -1]
X_in = inliers.drop('Strength', axis=1)
y_in = inliers['Strength']

kfold = KFold(n_splits=5, shuffle=True, random_state=23)
lr = LinearRegression()
res_in = cross_val_score(lr, X_in, y_in, cv=kfold)
print(res_in.mean())

X = concrete.drop('Strength', axis=1)
y = concrete['Strength']
res = cross_val_score(lr, X, y, cv=kfold)
print(res.mean())

############# Boston ###################
boston = pd.read_csv("Boston.csv")


clf = IsolationForest(contamination=0.05,
                      random_state=23)
clf.fit(boston)
predictions = clf.predict(boston)

inliers = boston[predictions != -1]
X_in = inliers.drop('medv', axis=1)
y_in = inliers['medv']

kfold = KFold(n_splits=5, shuffle=True, random_state=23)
lr = LinearRegression()
res_in = cross_val_score(lr, X_in, y_in, cv=kfold)
print(res_in.mean())

y = boston['medv']
X = boston.drop('medv', axis=1)
res = cross_val_score(lr, X, y, cv=kfold)
print(res.mean())