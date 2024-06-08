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
from sklearn.pipeline import Pipeline

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1).values
y = brupt['D'].values

params = {'n_neighbors': np.arange(1,21)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(knn, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################## Standard Scaler #####################
std_scl = StandardScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('SCL', std_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,21)}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################## MinMax Scaler #####################
mm_scl = MinMaxScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('SCL', mm_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,21)}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############ both scaling ###################

knn = KNeighborsClassifier()
pipe = Pipeline([('SCL', mm_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,21),
          'SCL':[mm_scl, std_scl]}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

pd_cv = pd.DataFrame( gcv.cv_results_ )

############### Glass Identification ################
glass = pd.read_csv("Glass.csv")
X = glass.drop('Type', axis=1)
y = glass['Type']


knn = KNeighborsClassifier()
pipe = Pipeline([('SCL', mm_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,31),
          'SCL':[mm_scl, std_scl]}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

## w/o scaling
knn = KNeighborsClassifier()
params = {'n_neighbors': np.arange(1,31)}
gcv = GridSearchCV(knn, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

#### all in one 
knn = KNeighborsClassifier()
pipe = Pipeline([('SCL', mm_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,31),
          'SCL':[mm_scl, std_scl, 'passthrough']}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############# Image Segmentation ###############
img_seg = pd.read_csv("Image_Segmentation.csv")
y = img_seg['Class'].values
X = img_seg.drop('Class', axis=1).values

knn = KNeighborsClassifier()
pipe = Pipeline([('SCL', mm_scl),('KNN', knn)])
print(pipe.get_params())
params = {'KNN__n_neighbors': np.arange(1,31),
          'SCL':[mm_scl, std_scl, 'passthrough']}
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)







