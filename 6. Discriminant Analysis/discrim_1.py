from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=23)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = lda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred = qda.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = qda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

############### Grid Search CV ################
lda = LinearDiscriminantAnalysis()
params = {'solver':['svd','lsqr','eigen']}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv = GridSearchCV(lda, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

qda = QuadraticDiscriminantAnalysis()
params = {'reg_param': np.linspace(0, 1, 10)}
gcv = GridSearchCV(qda, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


################ Glass Identification ##################
glass = pd.read_csv("Glass.csv")
y = glass['Type']
X = glass.drop('Type', axis=1)

lda = LinearDiscriminantAnalysis()
params = {'solver':['svd','lsqr','eigen']}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv_lda = GridSearchCV(lda, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_lda.fit(X, y)
print(gcv_lda.best_params_)
print(gcv_lda.best_score_)

qda = QuadraticDiscriminantAnalysis()
params = {'reg_param': np.linspace(0, 1, 10)}
gcv_qda = GridSearchCV(qda, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_qda.fit(X, y)
print(gcv_qda.best_params_)
print(gcv_qda.best_score_)

################ Satellite Imaging ####################
satellite = pd.read_csv("Satellite.csv", sep=";")
y = satellite['classes']
X = satellite.drop('classes', axis=1)

############ LDA
lda = LinearDiscriminantAnalysis()
params = {'solver':['svd','lsqr','eigen']}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv_lda = GridSearchCV(lda, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_lda.fit(X, y)
print(gcv_lda.best_params_)
print(gcv_lda.best_score_)   

############ QDA
qda = QuadraticDiscriminantAnalysis()
params = {'reg_param': np.linspace(0, 1, 10)}
gcv_qda = GridSearchCV(qda, param_grid=params,
                   cv=kfold, scoring='neg_log_loss')
gcv_qda.fit(X, y)
print(gcv_qda.best_params_)
print(gcv_qda.best_score_)

############## Logistic
params = {'penalty':[None, 'l1', 'l2','elasticnet'],
          'C': np.linspace(0,10,10),
          'l1_ratio': np.linspace(0, 1, 5) }
lr = LogisticRegression(solver='saga')
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)

gcv = GridSearchCV(lr, param_grid=params, cv=kfold,
                   verbose=3,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
