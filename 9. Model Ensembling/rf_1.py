from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=23)


rf = RandomForestClassifier(n_estimators=15,
                        random_state=23)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = rf.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

############### Grid search ###############

rf = RandomForestClassifier(random_state=23)
params = {'max_features':[3,5,6,7,10],
          'n_estimators': [25, 50, 100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv_rf = GridSearchCV(rf, param_grid=params,cv=kfold,
                       n_jobs=-1, scoring='neg_log_loss')
gcv_rf.fit(X, y)
print(gcv_rf.best_params_)
print(gcv_rf.best_score_)


