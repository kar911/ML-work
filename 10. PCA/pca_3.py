import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import StratifiedKFold

glass = pd.read_csv("../Cases/Glass Identification/Glass.csv")
X = glass.drop('Type', axis=1)
y = glass['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=23)
scaler = StandardScaler().set_output(transform='pandas')
prcomp = PCA().set_output(transform='pandas')

scl_trn = scaler.fit_transform(X_train)
trn_pca = prcomp.fit_transform(scl_trn)
print(np.cumsum(prcomp.explained_variance_ratio_*100))

c = 5
svm = SVC(kernel='linear')
svm.fit(trn_pca.iloc[:,:c], y_train)

scl_tst = scaler.transform(X_test)
tst_pca = prcomp.transform(scl_tst)
y_pred = svm.predict(tst_pca.iloc[:,:c])
print(accuracy_score(y_test, y_pred))

######## with pipeline 
scaler = StandardScaler().set_output(transform='pandas')
prcomp = PCA(n_components=5).set_output(transform='pandas')
svm = SVC(kernel='linear')

pipe = Pipeline([('SCL',scaler),('PCA',prcomp),('SVM',svm)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))

####################Grid Search############################
print(pipe.get_params())
params = {'SVM__C': np.linspace(0.001, 6, 20),
          'PCA__n_components':[4,5,6]}
scaler = StandardScaler().set_output(transform='pandas')
prcomp = PCA().set_output(transform='pandas')
svm = SVC(kernel='linear', probability=True,
          random_state=23)
pipe = Pipeline([('SCL',scaler),('PCA',prcomp),('SVM',svm)])
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_lin = GridSearchCV(pipe, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3,n_jobs=10)
gcv_lin.fit(X, y)
print(gcv_lin.best_params_)
print(gcv_lin.best_score_)
