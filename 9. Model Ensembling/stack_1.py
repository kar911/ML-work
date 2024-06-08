import pandas as pd
import numpy as np   
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=23)

svm = SVC(probability=True, random_state=23)
lr = LogisticRegression()
dtc = DecisionTreeClassifier(random_state=23)
gbm = GradientBoostingClassifier(random_state=23)

stack = StackingClassifier([('LR', lr),('SVM', svm),('TREE',dtc)],
                           passthrough=True,
                           final_estimator=gbm)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = stack.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

#############################################################
print(stack.get_params())
params = {'SVM__C':[0.5,1,1.5],
          'TREE__max_depth':[None, 3, 5],
          'final_estimator__learning_rate':[0.1, 0.5],
          'passthrough':[True, False]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)
gcv_stack = GridSearchCV(stack, param_grid=params,verbose=3, 
                         cv=kfold, scoring='neg_log_loss')
gcv_stack.fit(X, y)
print(gcv_stack.best_params_)
print(gcv_stack.best_score_)
