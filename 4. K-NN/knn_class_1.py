from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
kyp = pd.read_csv("Kyphosis.csv")
X = kyp.drop('Kyphosis', axis=1)
y = kyp['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=23)
## w/o scaling
scores = dict()
for k in [3,5,7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_prob = knn.predict_proba(X_test)
    scores[str(k)] = log_loss(y_test, y_pred_prob)
## Std Scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_trn_scl = scaler.transform(X_train)
X_tst_scl = scaler.transform(X_test)
scores_std = dict()
for k in [3,5,7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trn_scl, y_train)
    y_pred_prob = knn.predict_proba(X_tst_scl)
    scores_std[str(k)] = log_loss(y_test, y_pred_prob)
## mm scaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_trn_scl = scaler.transform(X_train)
X_tst_scl = scaler.transform(X_test)
scores_mm = dict()
for k in [3,5,7]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trn_scl, y_train)
    y_pred_prob = knn.predict_proba(X_tst_scl)
    scores_mm[str(k)] = log_loss(y_test, y_pred_prob)


################## Grid Search CV ###########################
params = {'n_neighbors': np.arange(1,11)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(knn, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)



