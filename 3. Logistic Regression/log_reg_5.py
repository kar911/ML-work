import pandas as pd
import numpy as np   
from sklearn.model_selection import (GridSearchCV,StratifiedKFold,
                                     train_test_split)
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (log_loss,confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

img = pd.read_csv("Image_Segmention.csv")
le = LabelEncoder()
img['Class'] = le.fit_transform(img['Class'])

X = img.drop('Class', axis=1)
y = img['Class']

params = {'penalty':[None, 'l1', 'l2','elasticnet'],
          'C': np.linspace(0,10,10),
          'l1_ratio': np.linspace(0, 1, 5) ,
          'multi_class':['ovr','multinomial']}
lr = LogisticRegression(random_state=23, solver='saga')
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)

gcv = GridSearchCV(lr, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

########## Inferencing
bm = gcv.best_estimator_
tst = pd.read_csv("tst_img.csv")
print(bm.predict(tst))
print(dict(zip(le.classes_, np.arange(0,7))))

prediction = le.inverse_transform(bm.predict(tst))
print(prediction)




