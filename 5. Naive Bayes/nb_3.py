import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

img = pd.read_csv("Image_Segmentation.csv")
le = LabelEncoder()
img['Class'] = le.fit_transform(img['Class'])

X = img.drop('Class', axis=1)
y = img['Class']

#### Naive Bayes
params = {'var_smoothing': np.linspace(1e-9, 10, 20)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)
nb = GaussianNB()
gcv = GridSearchCV(nb, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

#### K-NN
X = img.drop('Class', axis=1).values
y = img['Class'].values
mm_scl = MinMaxScaler()
std_scl = StandardScaler()
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

######### Logistic ##############

params = {'penalty':[None, 'l1', 'l2','elasticnet'],
          'C': np.linspace(0,10,10),
          'l1_ratio': np.linspace(0, 1, 5),
          'multi_class':['ovr','multinomial']}
lr = LogisticRegression(random_state=23, solver='saga')
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)

gcv = GridSearchCV(lr, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)



