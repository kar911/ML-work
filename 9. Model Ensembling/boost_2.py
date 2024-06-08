import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import KFold

housing = pd.read_csv("../Datasets/Housing.csv")
dum_hous = pd.get_dummies(housing, drop_first=True)
X = dum_hous.drop('price', axis=1)
y = dum_hous['price']

kfold = KFold(n_splits=5, shuffle=True,
                        random_state=23)

###################### GBM ##########################
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(random_state=23)
print(gbm.get_params())
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth':[1,3, 5, None],
          'n_estimators': [50, 100, 150]}
gcv_gbm = GridSearchCV(gbm, param_grid=params,
                       cv=kfold,verbose=3,n_jobs=10)
gcv_gbm.fit(X, y)
print(gcv_gbm.best_params_)
print(gcv_gbm.best_score_)

############### X G Boost ######################
from xgboost import XGBRegressor

x_gbm = XGBRegressor(random_state=23)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth':[1,3, 5, None],
          'n_estimators': [50, 100, 150]}
gcv_x_gbm = GridSearchCV(x_gbm, param_grid=params,
                      verbose=3,cv=kfold,n_jobs=10)
gcv_x_gbm.fit(X, y)
print(gcv_x_gbm.best_params_)
print(gcv_x_gbm.best_score_)

############### Light GBM ###################
from lightgbm import LGBMRegressor

l_gbm = LGBMRegressor(random_state=23)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth':[1,3, 5, None],
          'n_estimators': [50, 100, 150]}
gcv_l_gbm = GridSearchCV(l_gbm, param_grid=params,
                      verbose=3,cv=kfold,n_jobs=10)
gcv_l_gbm.fit(X, y)
print(gcv_l_gbm.best_params_)
print(gcv_l_gbm.best_score_)

############## Cat Boost ###################
from catboost import CatBoostRegressor
c_gbm = CatBoostRegressor(random_state=23)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth':[1,3, 5, None],
          'n_estimators': [50, 100, 150]}
gcv_c_gbm = GridSearchCV(c_gbm, param_grid=params,
                      verbose=3,cv=kfold,n_jobs=10)
gcv_c_gbm.fit(X, y)
print('CAT')
print(gcv_c_gbm.best_params_)
print(gcv_c_gbm.best_score_)

#### w/o hot encoding

X = housing.drop('price', axis=1)
y = housing['price']

all_cats = list(X.dtypes[X.dtypes==object].index)
c_gbm = CatBoostRegressor(random_state=23)
#c_gbm.fit(X, y, cat_features=all_cats)

params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth':[1,3, 5, None],
          'n_estimators': [50, 100, 150],
          'cat_features':[all_cats]}
gcv_c_gbm = GridSearchCV(c_gbm, param_grid=params,
                      verbose=3,cv=kfold,n_jobs=10)
gcv_c_gbm.fit(X, y)

print('CAT_OHE')
print(gcv_c_gbm.best_params_)
print(gcv_c_gbm.best_score_)
print('Gradi_B_')
print(gcv_gbm.best_params_)
print(gcv_gbm.best_score_)
print('Xg')
print(gcv_x_gbm.best_params_)
print(gcv_x_gbm.best_score_)
print('Light')
print(gcv_l_gbm.best_params_)
print(gcv_l_gbm.best_score_)
