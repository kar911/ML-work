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

glass = pd.read_csv("Glass.csv")
y = glass['Type']
X = glass.drop('Type', axis=1)

le = LabelEncoder()
le_y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, le_y, 
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=23)
lr = LogisticRegression(penalty=None,multi_class='ovr', 
                        solver='saga', l1_ratio=0.5)
lr.fit(X_train, y_train)
y_pred_prob = lr.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot()
plt.xticks(rotation=90)
plt.show()

print(classification_report(y_test, y_pred))
print(dict(zip(le.classes_, np.arange(0,6))))

###################################################
params = {'penalty':[None, 'l1', 'l2','elasticnet'],
          'C': np.linspace(0,10,10),
          'l1_ratio': np.linspace(0, 1, 5) ,
          'multi_class':['ovr','multinomial'],
          'solver':['sag','saga','lbfgs','liblinear',
                    'newton-cg','newton-cholesky']}
lr = LogisticRegression(random_state=23)
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)

gcv = GridSearchCV(lr, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
#{'C': 1.1111111111111112, 'l1_ratio': 0.0, 'multi_class': 'multinomial', 'penalty': 'l2', 'solver': 'newton-cg'}
best_model = gcv.best_estimator_
######## Serialization ##################
from joblib import dump 
dump(best_model, 'logreg.joblib') 


#### Deserialization
from joblib import load 
bm = load("logreg.joblib")

#### Loading unlabeled
tst = pd.read_csv("tst_Glass.csv")
print(bm.predict(tst))
