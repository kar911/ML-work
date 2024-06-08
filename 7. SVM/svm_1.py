from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=23)

svm = SVC(kernel='linear', C = 0.5, probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = svm.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

################## Grid Search $$$$$$$$$$$$$$$$$$

### Linear
params = {'C': np.linspace(0.001, 6, 20)}
svm = SVC(kernel='linear', probability=True,
          random_state=23)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_lin = GridSearchCV(svm, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3)
gcv_lin.fit(X, y)
print(gcv_lin.best_params_)
print(gcv_lin.best_score_)

### Poly
params = {'C': np.linspace(0.001, 6, 20), 
          'degree': [2,3,4,5,6],
          'coef0': np.linspace(-1, 2, 10)}
svm = SVC(kernel='poly', probability=True,
          random_state=23)
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)
gcv_poly = GridSearchCV(svm, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3)
gcv_poly.fit(X, y)
print(gcv_poly.best_params_)
print(gcv_poly.best_score_)

### Radial
params = {'C': np.linspace(0.001, 6, 20), 
          'gamma': np.linspace(0.001, 5, 10)}
svm = SVC(kernel='rbf', probability=True,
          random_state=23)
gcv_rbf = GridSearchCV(svm, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3)
gcv_rbf.fit(X, y)
print(gcv_rbf.best_params_)
print(gcv_rbf.best_score_)

############## with Scaling #######################
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
## Linear
svm_lin = SVC(kernel='linear', probability=True,
          random_state=23)
pipe_lin = Pipeline([('SCL', None),('SVM', svm_lin)])
params = {'SVM__C': np.linspace(0.001, 6, 20),
          'SCL': [None, std_scaler, mm_scaler]}
gcv_lin = GridSearchCV(pipe_lin, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3)
gcv_lin.fit(X, y)
print(gcv_lin.best_params_)
print(gcv_lin.best_score_)

## Polynomial
svm_poly = SVC(kernel='poly', probability=True,
          random_state=23)
pipe_poly = Pipeline([('SCL', None),('SVM', svm_poly)])
params = {'SVM__C': np.linspace(0.001, 6, 20), 
          'SVM__degree': [2,3,4,5,6],
          'SVM__coef0': np.linspace(-1, 2, 10),
          'SCL': [None, std_scaler, mm_scaler]}
gcv_poly = GridSearchCV(pipe_poly, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3)
gcv_poly.fit(X, y)
print(gcv_poly.best_params_)
print(gcv_poly.best_score_)

## Radial
svm_rbf = SVC(kernel='rbf', probability=True,
          random_state=23)
pipe_rbf = Pipeline([('SCL', None),('SVM', svm_rbf)])
params = {'SVM__C': np.linspace(0.001, 6, 20), 
          'SVM__gamma': np.linspace(0.001, 5, 10),
          'SCL': [None, std_scaler, mm_scaler]}
gcv_rbf = GridSearchCV(pipe_rbf, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3)
gcv_rbf.fit(X, y)
print(gcv_rbf.best_params_)
print(gcv_rbf.best_score_)

################ Kyphosis #######################
kyph = pd.read_csv("Kyphosis.csv")
y = kyph['Kyphosis']
X = kyph.drop('Kyphosis', axis=1)

std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
## Linear
svm_lin = SVC(kernel='linear', probability=True,
          random_state=23)
pipe_lin = Pipeline([('SCL', None),('SVM', svm_lin)])
params = {'SVM__C': np.linspace(0.001, 6, 20),
          'SCL': [None, std_scaler, mm_scaler]}
gcv_lin = GridSearchCV(pipe_lin, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3)
gcv_lin.fit(X, y)
print(gcv_lin.best_params_)
print(gcv_lin.best_score_)

## Polynomial
svm_poly = SVC(kernel='poly', probability=True,
          random_state=23)
pipe_poly = Pipeline([('SCL', None),('SVM', svm_poly)])
params = {'SVM__C': np.linspace(0.001, 6, 20), 
          'SVM__degree': [2,3,4,5,6],
          'SVM__coef0': np.linspace(-1, 2, 10),
          'SCL': [None, std_scaler, mm_scaler]}
gcv_poly = GridSearchCV(pipe_poly, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3)
gcv_poly.fit(X, y)
print(gcv_poly.best_params_)
print(gcv_poly.best_score_)

## Radial
svm_rbf = SVC(kernel='rbf', probability=True,
          random_state=23)
pipe_rbf = Pipeline([('SCL', None),('SVM', svm_rbf)])
params = {'SVM__C': np.linspace(0.001, 6, 20), 
          'SVM__gamma': np.linspace(0.001, 5, 10),
          'SCL': [None, std_scaler, mm_scaler]}
gcv_rbf = GridSearchCV(pipe_rbf, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3)
gcv_rbf.fit(X, y)
print(gcv_rbf.best_params_)
print(gcv_rbf.best_score_)
