import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']

dtc = DecisionTreeClassifier(random_state=23, max_depth=1)
lr = LogisticRegression()
svm = SVC(probability=True, random_state=23)

ada = AdaBoostClassifier(random_state=23)
params = {'estimator':[dtc, lr, svm],
          'n_estimators':[25,50,100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv_ada = GridSearchCV(ada, param_grid=params,
                       cv=kfold, scoring='neg_log_loss')
gcv_ada.fit(X, y)
print(gcv_ada.best_params_)
print(gcv_ada.best_score_)

###################### GBM ##########################
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(random_state=23)
print(gbm.get_params())
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth':[1,3, 5, None],
          'n_estimators': [50, 100, 150]}
gcv_gbm = GridSearchCV(gbm, param_grid=params,
                       cv=kfold, scoring='neg_log_loss')
gcv_gbm.fit(X, y)
print(gcv_gbm.best_params_)
print(gcv_gbm.best_score_)

############### X G Boost ######################
from xgboost import XGBClassifier

x_gbm = XGBClassifier(random_state=23)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth':[1,3, 5, None],
          'n_estimators': [50, 100, 150]}
gcv_x_gbm = GridSearchCV(x_gbm, param_grid=params,
                      verbose=3,cv=kfold, scoring='neg_log_loss')
gcv_x_gbm.fit(X, y)
print(gcv_x_gbm.best_params_)
print(gcv_x_gbm.best_score_)

############### Light GBM ###################
from lightgbm import LGBMClassifier

l_gbm = LGBMClassifier(random_state=23)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth':[1,3, 5, None],
          'n_estimators': [50, 100, 150]}
gcv_l_gbm = GridSearchCV(l_gbm, param_grid=params,
                      verbose=3,cv=kfold, scoring='neg_log_loss')
gcv_l_gbm.fit(X, y)
print(gcv_l_gbm.best_params_)
print(gcv_l_gbm.best_score_)

############## Cat Boost ###################
from catboost import CatBoostClassifier
c_gbm = CatBoostClassifier(random_state=23)
params = {'learning_rate':np.linspace(0.001, 1, 10),
          'max_depth':[1,3, 5, None],
          'n_estimators': [50, 100, 150]}
gcv_c_gbm = GridSearchCV(c_gbm, param_grid=params,
                      verbose=3,cv=kfold, scoring='neg_log_loss')
gcv_c_gbm.fit(X, y)
print(gcv_c_gbm.best_params_)
print(gcv_c_gbm.best_score_)



