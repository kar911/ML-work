import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error as mse
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np 
from pmdarima.arima import auto_arima 

df = pd.read_csv("monthly-milk-production-pounds-p.csv")

y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]

model = auto_arima(y_train, trace=True,
                   error_action='ignore', 
                   suppress_warnings=True,
                   seasonal=True,m=12)

forecast = model.predict(n_periods=len(y_test))

#plot the predictions for validation set
plt.plot(y_train, label='Train',color="blue")
plt.plot(y_test, label='Valid',color="pink")
plt.plot(forecast, label='Prediction',color="purple")
error = round(sqrt(mse(y_test, forecast)),2)
plt.text(100,600, "RMSE="+str(error))
plt.legend(loc='best')
plt.show()

######## WGEM ##############
wgem = pd.read_csv("WGEM-IND_CPTOTNSXN.csv")

y_train = wgem['Value'][:-4]
y_test = wgem['Value'][-4:]

model = auto_arima(y_train, trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   seasonal=True,m=1)

forecast = model.predict(n_periods=len(y_test))

#plot the predictions for validation set
plt.plot(y_train, label='Train',color="blue")
plt.plot(y_test, label='Valid',color="pink")
plt.plot(forecast, label='Prediction',color="purple")
error = round(sqrt(mse(y_test, forecast)),2)
plt.text(5,80, "RMSE="+str(error))
plt.legend(loc='best')
plt.show()

################ BUNDESBANK
bund = pd.read_csv("BUNDESBANK-BBK01_WT5511.csv")
y_train = bund['Value'][:-6]
y_test = bund['Value'][-6:]

model = auto_arima(y_train, trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   seasonal=True,m=12)

forecast = model.predict(n_periods=len(y_test))

#plot the predictions for validation set
plt.plot(y_train, label='Train',color="blue")
plt.plot(y_test, label='Valid',color="pink")
plt.plot(forecast, label='Prediction',color="purple")
error = round(sqrt(mse(y_test, forecast)),2)
plt.text(100,600, "RMSE="+str(error))
plt.legend(loc='best')
plt.show()


