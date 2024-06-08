from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay 
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=23)

lr = LogisticRegression(penalty='l1', 
                        solver='saga', l1_ratio=0.5)
lr.fit(X_train, y_train)
print(lr.coef_)


# Outcomes with biggest probability
y_pred = lr.predict(X_test)
print(y_pred)

print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Working','Left'])
disp.plot()
plt.show()

print(classification_report(y_test, y_pred))

################# ROC Curve #########################
from sklearn.metrics import roc_curve, roc_auc_score
lr = LogisticRegression(penalty=None, 
                        solver='saga', l1_ratio=0.5)
lr.fit(X_train, y_train)
y_pred_prob = lr.predict_proba(X_test)
y_pred_prob = y_pred_prob[:,1]

fpr, tpr, thres = roc_curve(y_test, y_pred_prob)
plt.scatter(fpr, tpr, c='red')
plt.plot(fpr, tpr)
plt.xlabel("1 - Specificity")
plt.ylabel("Sensitivity")
plt.show()

print(roc_auc_score(y_test, y_pred_prob))

############ Grid Search CV ################

params = {'penalty':[None, 'l1', 'l2','elasticnet'],
          'l1_ratio':[0.25, 0.5, 0.75]}
lr = LogisticRegression(solver='saga')
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)

## Accuracy Score
gcv = GridSearchCV(lr, param_grid=params, cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

## ROC AUC
gcv = GridSearchCV(lr, param_grid=params, cv=kfold,
                   scoring='roc_auc')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

## F1 Score
params = {'penalty':[None, 'l1', 'l2','elasticnet'],
          'l1_ratio':[0.25, 0.5, 0.75]}
gcv = GridSearchCV(lr, param_grid=params, cv=kfold,
                   scoring='f1_macro')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)



