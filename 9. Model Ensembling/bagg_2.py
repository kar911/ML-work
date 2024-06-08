import pandas as pd
import numpy as np
from sklearn.linear_model import (LinearRegression,
    ElasticNet)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import (GridSearchCV,
                KFold)
concrete = pd.read_csv("../Cases/Concrete Strength/Concrete_Data.csv")

y = concrete['Strength']
X = concrete.drop('Strength', axis=1)

kfold = KFold(n_splits=5, shuffle=True, 
              random_state=23)
bagg = BaggingRegressor(random_state=23)
dtr = DecisionTreeRegressor(random_state=23)
knn = KNeighborsRegressor()
elastic = ElasticNet()
lr = LinearRegression()
params = {'estimator':[lr, elastic,knn,dtr],
          'n_estimators':[10, 50, 75]}
gcv_bgg = GridSearchCV(bagg, param_grid=params,
                       cv=kfold, n_jobs=10,verbose=0)
gcv_bgg.fit(X, y)
c=gcv_bgg.best_params_
v=gcv_bgg.best_score_

########## Single estimator #############
bagg = BaggingRegressor(random_state=23,
                        estimator=dtr)
params = {'estimator__max_depth':[None, 3, 5],
          'estimator__min_samples_split':[2, 5, 10],
          'estimator__min_samples_leaf':[1, 5, 10],
          'n_estimators':[10, 50, 75]}
gcv_bgg = GridSearchCV(bagg, param_grid=params,
                       cv=kfold, n_jobs=10,verbose=0)
gcv_bgg.fit(X, y)
print('Single_esti')
print(gcv_bgg.best_params_)
print(gcv_bgg.best_score_)
print('Mul_esti')
print(c)
print(v)
