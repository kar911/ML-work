import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
hr = pd.read_csv("HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y,
                                                    random_state=23)

dtc = DecisionTreeClassifier(random_state=23,
                             max_depth=None)
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = dtc.predict_proba(X_test)
print(log_loss(y_test,y_pred_prob))

###########################################
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'max_depth':[2,3,4,5,6,7,None],
          'min_samples_split':[2, 5, 10, 20],
          'min_samples_leaf':[1,5,7,10,20]}
gcv_tree = GridSearchCV(dtc, param_grid=params,
                        cv=kfold, verbose=3,
                        scoring='neg_log_loss')
gcv_tree.fit(X, y)
print(gcv_tree.best_params_)
print(gcv_tree.best_score_)
pd_cv = pd.DataFrame(gcv_tree.cv_results_)

#### Best Tree
best_tree = gcv_tree.best_estimator_
plt.figure(figsize=(35,15))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=['0','1'],
               filled=True,fontsize=12)
plt.show() 


importances = best_tree.feature_importances_
pd_imp = pd.DataFrame({'Feature':list(X.columns),
                       'Importance':importances})
pd_imp.sort_values(by='Importance', inplace=True)
plt.barh(pd_imp['Feature'], pd_imp['Importance'])
plt.title("Feature Importances Plot")
plt.show()


################# Bankrutpcy ###############

brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'max_depth':[2,3,4,5,6,7,None],
          'min_samples_split':[2, 5, 10, 20],
          'min_samples_leaf':[1,5,7,10,20]}
gcv_tree = GridSearchCV(dtc, param_grid=params,
                        cv=kfold, verbose=3,
                        scoring='neg_log_loss')
gcv_tree.fit(X, y)
print(gcv_tree.best_params_)
print(gcv_tree.best_score_)
pd_cv = pd.DataFrame(gcv_tree.cv_results_)

#### Best Tree
best_tree = gcv_tree.best_estimator_
plt.figure(figsize=(35,15))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=['0','1'],
               filled=True,fontsize=20)
plt.show() 

importances = best_tree.feature_importances_
pd_imp = pd.DataFrame({'Feature':list(X.columns),
                       'Importance':importances})
pd_imp.sort_values(by='Importance', inplace=True)
plt.barh(pd_imp['Feature'], pd_imp['Importance'])
plt.title("Feature Importances Plot")
plt.show()


############### Glass Identification ################

glass = pd.read_csv("Glass.csv")
X = glass.drop('Type', axis=1)
y = glass['Type']
le = LabelEncoder()
le_y = le.fit_transform(y)
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'max_depth':[2,3,4,5,6,7,None],
          'min_samples_split':[2, 5, 10, 20],
          'min_samples_leaf':[1,5,7,10,20]}
gcv_tree = GridSearchCV(dtc, param_grid=params,
                        cv=kfold, verbose=3,
                        scoring='neg_log_loss')
gcv_tree.fit(X, le_y)
print(gcv_tree.best_params_)
print(gcv_tree.best_score_)
pd_cv = pd.DataFrame(gcv_tree.cv_results_)

#### Best Tree
best_tree = gcv_tree.best_estimator_
plt.figure(figsize=(35,15))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=list(le.classes_),
               filled=True,fontsize=20)
plt.show() 

importances = best_tree.feature_importances_
pd_imp = pd.DataFrame({'Feature':list(X.columns),
                       'Importance':importances})
pd_imp.sort_values(by='Importance', inplace=True)
plt.barh(pd_imp['Feature'], pd_imp['Importance'])
plt.title("Feature Importances Plot")
plt.show()

################################################
iris = pd.read_csv("iris.csv")
X = iris.drop('Species', axis=1)
y = iris['Species']

le = LabelEncoder()
le_y = le.fit_transform(y)
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'max_depth':[2,3,4,5,6,7,None],
          'min_samples_split':[2, 5, 10, 20],
          'min_samples_leaf':[1,5,7,10,20]}
gcv_tree = GridSearchCV(dtc, param_grid=params,
                        cv=kfold, verbose=3,
                        scoring='neg_log_loss')
gcv_tree.fit(X, le_y)
print(gcv_tree.best_params_)
print(gcv_tree.best_score_)
pd_cv = pd.DataFrame(gcv_tree.cv_results_)

#### Best Tree
best_tree = gcv_tree.best_estimator_
plt.figure(figsize=(35,15))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=list(le.classes_),
               filled=True,fontsize=20)
plt.show() 

importances = best_tree.feature_importances_
pd_imp = pd.DataFrame({'Feature':list(X.columns),
                       'Importance':importances})
pd_imp.sort_values(by='Importance', inplace=True)
plt.barh(pd_imp['Feature'], pd_imp['Importance'])
plt.title("Feature Importances Plot")
plt.show()