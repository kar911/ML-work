import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet

housing = pd.read_csv("../Datasets/Housing.csv")
dum_hous = pd.get_dummies(housing, drop_first=True)
X = dum_hous.drop('price', axis=1)
y = dum_hous['price']

lr = LinearRegression()
dtr = DecisionTreeRegressor(random_state=23)
elastic = ElasticNet()
rf = RandomForestRegressor(random_state=23)

stack = StackingRegressor([('LR',lr),('TREE',dtr),('EL',elastic)],
                          final_estimator=rf)
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=23)

params = {'EL__l1_ratio':[0.001, 0.2, 0.5],
          'EL__alpha':[0.01, 0.5, 1],
          'TREE__max_depth': [None, 3],
          'passthrough':[True, False],
          'final_estimator__max_features':[2,3,4,5]}
gcv_st = GridSearchCV(stack, param_grid=params,
                       cv=kfold,verbose=3,n_jobs=10)
gcv_st.fit(X, y)
print(gcv_st.best_params_)
print(gcv_st.best_score_)

# {'EL__alpha': 1, 'EL__l1_ratio': 0.5, 'TREE__max_depth': 3, 'final_estimator__max_features': 2, 'passthrough': True}
# 0.646516905255689