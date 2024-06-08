import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']

scaler = MinMaxScaler().set_output(transform='pandas')

sgd = SGDClassifier(random_state=23, loss='log_loss')
pipe = Pipeline([('SCL', scaler),('SDG', sgd)])
params = {'SDG__penalty':['l1','l2','elasticnet',None],
          'SDG__eta0':[0.01, 0.2, 0.3],
          'SDG__learning_rate':['constant',
                'optimal','invscaling','adaptive']}
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)






