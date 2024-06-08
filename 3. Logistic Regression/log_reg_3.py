from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

kyp = pd.read_csv("Kyphosis.csv")
X = kyp.drop('Kyphosis', axis=1)
y = kyp['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=23)

lr = LogisticRegression(penalty=None, 
                        solver='saga', l1_ratio=0.5)
lr.fit(X_train, y_train)
y_pred_prob = lr.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

############################################################
params = {'penalty':[None, 'l1', 'l2','elasticnet'],
          'C': np.linspace(0,10,10),
          'l1_ratio': np.linspace(0, 10, 5) }
lr = LogisticRegression(solver='saga')
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)

gcv = GridSearchCV(lr, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_cv = pd.DataFrame(gcv.cv_results_ )

############## Sonar ###############################
sonar = pd.read_csv("Sonar.csv")
X = sonar.drop('Class', axis=1)
y = sonar['Class']

params = {'penalty':[None, 'l1', 'l2','elasticnet'],
          'C': np.linspace(0,10,10),
          'l1_ratio': np.linspace(0, 10, 5) }
lr = LogisticRegression(solver='saga')
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)

gcv = GridSearchCV(lr, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)



