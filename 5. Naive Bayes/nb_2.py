from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size=0.3,stratify=y,
                                   random_state=23)
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_prob = nb.predict_proba(X_test)
y_pred = nb.predict(X_test)
#print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))

################ Grid Search CV ########
params = {'var_smoothing': np.linspace(1e-9, 10, 20)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)
nb = GaussianNB()
gcv = GridSearchCV(nb, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


