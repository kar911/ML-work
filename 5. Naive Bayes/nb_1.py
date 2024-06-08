import pandas as pd
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss 
from sklearn.model_selection import train_test_split 
tel = pd.read_csv("tel_bayes.csv")
dum_tel = pd.get_dummies(tel, drop_first=True)
nb = BernoulliNB(alpha=0, force_alpha=True)

X = dum_tel[['TT_gt_100_y' , 'Gender_male']]
y = dum_tel['Response_not bought']

nb.fit(X, y) # Calculates apriori probs

tst = np.array([["n","female"],
                ["n","male"],
                ["y","male"],
                ["y", "female"]])
tst = pd.DataFrame(tst, columns = ['TT_gt_100',
                                   'Gender'])
dum_tst = pd.get_dummies(tst, drop_first=True)

nb.predict_proba(dum_tst)
# P(True)  P(False)

#####################################################
telecom = pd.read_csv("Telecom.csv")
dum_telecom = pd.get_dummies(telecom, drop_first=True)
nb = BernoulliNB(alpha=1.5)

X = dum_telecom.drop('Response_Y', axis=1)
y = dum_telecom['Response_Y']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=23)
nb.fit(X_train, y_train)
y_pred_prob = nb.predict_proba(X_test)
y_pred = nb.predict(X_test)
#print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))

################ Grid Search CV ########
params = {'alpha': np.linspace(0, 3, 10)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)
nb = BernoulliNB()
gcv = GridSearchCV(nb, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############# Cancer #############################
cancer = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Cancer\Cancer.csv",
                     index_col=0)
dum_canc = pd.get_dummies(cancer, drop_first=True)
X = dum_canc.drop('Class_recurrence-events', axis=1)
y = dum_canc['Class_recurrence-events']

params = {'alpha': np.linspace(0.001, 10, 20)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)
nb = BernoulliNB()
gcv = GridSearchCV(nb, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

######### K-NN ##############
from sklearn.neighbors import KNeighborsClassifier
X = dum_canc.drop('Class_recurrence-events', axis=1).values
y = dum_canc['Class_recurrence-events'].values
params = {'n_neighbors': np.arange(1,31)}

kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)
knn = KNeighborsClassifier()
gcv = GridSearchCV(knn, param_grid=params, cv=kfold, 
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

######### Logistic ##############
from sklearn.linear_model import LogisticRegression
params = {'penalty':[None, 'l1', 'l2','elasticnet'],
          'C': np.linspace(0,10,10),
          'l1_ratio': np.linspace(0, 1, 5)}
lr = LogisticRegression(random_state=23, solver='saga')
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=23)

gcv = GridSearchCV(lr, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)



