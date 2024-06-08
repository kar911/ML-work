 from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import KFold, cross_val_score 
from sklearn.model_selection import GridSearchCV 

chem = pd.read_csv("ChemicalProcess.csv")
X = chem.drop('Yield', axis=1)
y = chem['Yield']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=23)
imp = SimpleImputer(strategy='mean').set_output(transform="pandas")
imp.fit(X_train)
X_trn_imp = imp.transform(X_train)
X_tst_imp = imp.transform(X_test)

elastic = ElasticNet(alpha=0.001, l1_ratio=0.5)
elastic.fit(X_trn_imp, y_train)
y_pred = elastic.predict(X_tst_imp)
print(r2_score(y_test, y_pred))

######### Using Pipeline ##############
from sklearn.pipeline import Pipeline

pipe = Pipeline([('IMP', imp), ('ELASTIC', elastic)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))

########### K-Fold CV ################

kfold = KFold(n_splits=5, shuffle=True, random_state=23)
results = cross_val_score(pipe, X, y, cv=kfold)
print(results.mean())

################## Grid Search CV ###############
imp = SimpleImputer().set_output(transform="pandas")
pipe = Pipeline([('IMP', imp), ('ELASTIC', elastic)])
print(pipe.get_params())
params = {'ELASTIC__alpha': np.linspace(0.001, 10, 20) , 
          'ELASTIC__l1_ratio': np.linspace(0,1, 10),
          'IMP__strategy':['mean', 'median']}
gcv = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv.fit(X,y)
print(gcv.best_score_)
print(gcv.best_params_)


best_el = gcv.best_estimator_
print(elastic.coef_)
print(elastic.intercept_)





