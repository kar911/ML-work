from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor 

housing = pd.read_csv("Housing.csv")
dum_hous = pd.get_dummies(housing, drop_first=True)
X = dum_hous.drop('price', axis=1)
y = dum_hous['price']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=23)
knn = KNeighborsRegressor()
elastic = ElasticNet()
dtr = DecisionTreeRegressor(random_state=23)

voting = VotingRegressor(estimators=[('KNN',knn),
                                      ('ELASTIC',elastic),
                                      ('TREE', dtr)])
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(r2_score(y_test, y_pred))


##################################################
voting = VotingRegressor(estimators=[('KNN',knn),
                                     ('ELASTIC',elastic),
                                     ('TREE', dtr)])
#print(voting.get_params())
params = {'KNN__n_neighbors': [2,5,7],
          'ELASTIC__l1_ratio': np.linspace(0.001, 5, 5),
          'ELASTIC__alpha': np.linspace(0.001, 6, 5),
          'TREE__max_depth':[3, 5 ,None]}

kfold = KFold(n_splits=5, shuffle=True, random_state=23)
gcv_vot = GridSearchCV(voting, param_grid=params, cv=kfold,
                        n_jobs=-1)
gcv_vot.fit(X, y)
print(gcv_vot.best_params_)
print(gcv_vot.best_score_)

