import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score, log_loss

le = LabelEncoder()

tiny = pd.read_csv("tinydata.csv", index_col=0)
X = tiny[['Fat','Salt']]
y = le.fit_transform( tiny['Acceptance'] )
mlp = MLPClassifier( hidden_layer_sizes=(3,),
    random_state=23 )
mlp.fit(X, y)
print(mlp.coefs_) # weights
print(mlp.intercepts_) # biases

############## Bankruptcy ##################
brupt = pd.read_csv("Bankruptcy.csv")
X = brupt.drop(['NO', 'YR', 'D'], axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   stratify=y,
                                   random_state=23)
mlp = MLPClassifier( hidden_layer_sizes=(10,5,),
    random_state=23, activation='logistic')
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = mlp.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

## scaling

scaler = MinMaxScaler().set_output(transform='pandas')
mlp = MLPClassifier( hidden_layer_sizes=(10,5,),
    random_state=23, activation='logistic', batch_size=10)
pipe = Pipeline([('SCL', scaler),('MLP', mlp)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = pipe.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))
