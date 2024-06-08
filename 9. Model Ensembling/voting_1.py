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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=23)

svm_l = SVC(kernel='linear', probability=True, 
            random_state=23)
svm_r = SVC(kernel='rbf', probability=True, 
            random_state=23)
lr = LogisticRegression()

voting = VotingClassifier(estimators=[('LIN_SVM',svm_l),
                                      ('RBF_SVM',svm_r),
                                      ('LR', lr)],
                          voting='soft')
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = voting.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

##################################################
voting = VotingClassifier(estimators=[('LIN_SVM',svm_l),
                                      ('RBF_SVM',svm_r),
                                      ('LR', lr)],
                          voting='soft',
                          weights=[0.25, 0.25, 0.5])
#print(voting.get_params())
params = {'LIN_SVM__C': np.linspace(0.001, 6, 5),
          'RBF_SVM__gamma': np.linspace(0.001, 5, 5),
          'RBF_SVM__C': np.linspace(0.001, 6, 5),
          'LR__penalty':['l1','l2',None]}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
gcv_vot = GridSearchCV(voting, param_grid=params, cv=kfold,
                       scoring='neg_log_loss', n_jobs=-1)
gcv_vot.fit(X, y)
print(gcv_vot.best_params_)
print(gcv_vot.best_score_)

