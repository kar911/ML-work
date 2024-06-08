from sklearn.metrics import mean_squared_error as mse 
import os
os.chdir("C:/Training/Academy/Statistics (Python)/Datasets")
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from numpy import sqrt

wgem = pd.read_csv("WGEM-IND_CPTOTNSXN.csv")

y = wgem['Value']
y_train = y.iloc[:-4]
y_test = y.iloc[-4:]

alpha = 0.8
# Simple Exponential Smoothing
from statsmodels.tsa.api import SimpleExpSmoothing
ses = SimpleExpSmoothing(y_train)
fit1 = ses.fit(smoothing_level=alpha)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

print(fit1.params)
print("RMSE =",sqrt(mse(y_test, fcast1)))

########## gasoline ################
gas = pd.read_csv("Gasoline.csv")
y = gas['Sales']
y_train = y.iloc[:-4]
y_test = y.iloc[-4:]


alpha = 0.3
# Simple Exponential Smoothing
from statsmodels.tsa.api import SimpleExpSmoothing
ses = SimpleExpSmoothing(y_train)
fit1 = ses.fit(smoothing_level=alpha)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

print(fit1.params)
print("RMSE =",sqrt(mse(y_test, fcast1)))

