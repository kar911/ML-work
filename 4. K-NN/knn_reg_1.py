from sklearn.metrics import r2_score, mean_squared_error as mse
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

boston = pd.read_csv("Boston.csv")
y = boston['medv'].values
X = boston.drop('medv', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23)
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(mse(y_test, y_pred))
print(r2_score(y_test, y_pred))

#################### Grid Search CV ###########################

params = {'n_neighbors': np.arange(1,31)}
kfold = KFold(n_splits=5, shuffle=True, 
                        random_state=23)
knn = KNeighborsRegressor()
gcv = GridSearchCV(knn, param_grid=params, cv=kfold, 
                   scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

######### with all scaling ########
std_scl = StandardScaler()
mm_scl = MinMaxScaler()
knn = KNeighborsRegressor()
pipe = Pipeline([('SCL', mm_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,31),
          'SCL':[mm_scl, std_scl, 'passthrough']}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############### Randomized Search CV ######################
from sklearn.model_selection import RandomizedSearchCV

std_scl = StandardScaler()
mm_scl = MinMaxScaler()
knn = KNeighborsRegressor()
pipe = Pipeline([('SCL', mm_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,101),
          'SCL':[mm_scl, std_scl, 'passthrough']}
rgcv = RandomizedSearchCV(pipe, param_distributions=params,
                          n_iter=60, random_state=23)
rgcv.fit(X, y)
print(rgcv.best_params_)
print(rgcv.best_score_)

pd_cv = pd.DataFrame( rgcv.cv_results_ )  


