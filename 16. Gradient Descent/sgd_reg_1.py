from sklearn.linear_model import SGDRegressor
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from sklearn.pipeline import Pipeline

pizza = pd.read_csv("pizza.csv")
scaler = MinMaxScaler().set_output(transform='pandas')

sgd = SGDRegressor(random_state=23, penalty=None,
                   eta0=0.2, learning_rate='constant')
pipe = Pipeline([('SCL', scaler),('SDG', sgd)])

X = pizza[['Promote']]
y = pizza['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   random_state=23) 

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))
