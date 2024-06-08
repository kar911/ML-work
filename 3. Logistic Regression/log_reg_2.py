from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import StratifiedKFold

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']

params = {'penalty':[None, 'l1', 'l2','elasticnet'],
          'l1_ratio':[0.25, 0.5, 0.75]}
lr = LogisticRegression(solver='saga')
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)

gcv = GridSearchCV(lr, param_grid=params, cv=kfold,
                   scoring='roc_auc')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
