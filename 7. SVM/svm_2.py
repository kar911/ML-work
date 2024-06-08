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


glass = pd.read_csv("Glass.csv")
y = glass['Type']
X = glass.drop('Type', axis=1)

std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
## Linear
svm_lin = SVC(kernel='linear', probability=True,
          random_state=23)
pipe_lin = Pipeline([('SCL', None),('SVM', svm_lin)])
params = {'SVM__C': np.linspace(0.001, 6, 20),
          'SVM__decision_function_shape':['ovo','ovr'],
          'SCL': [None, std_scaler, mm_scaler]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)
gcv_lin = GridSearchCV(pipe_lin, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', verbose=3)
gcv_lin.fit(X, y)
print(gcv_lin.best_params_)
print(gcv_lin.best_score_)

