import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

concrete = pd.read_csv("Concrete_Data.csv")
X = concrete.drop('Strength', axis=1).values
y = concrete['Strength'].values

kfold = KFold(n_splits=5, shuffle=True, 
                        random_state=23)
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
