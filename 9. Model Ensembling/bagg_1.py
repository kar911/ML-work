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
from sklearn.ensemble import BaggingClassifier

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

bag = BaggingClassifier(lr, n_estimators=15,
                        random_state=23)

bag.fit(X_train, y_train)
y_pred = bag.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = bag.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

############### Grid search ###############
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
dtc = DecisionTreeClassifier(random_state=23)
bag = BaggingClassifier(random_state=23)
params = {'estimator':[svm_l, svm_r, lr,
                       lda, qda, dtc]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
gcv_bgg = GridSearchCV(bag, param_grid=params,cv=kfold,
                       n_jobs=-1, scoring='neg_log_loss')
gcv_bgg.fit(X, y)
print(gcv_bgg.best_params_)
print(gcv_bgg.best_score_)


