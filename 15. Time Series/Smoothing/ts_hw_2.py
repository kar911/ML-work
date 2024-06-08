from sklearn.metrics import mean_squared_error as mse 
import os
os.chdir("C:/Training/Academy/Statistics (Python)/Datasets")
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from numpy import sqrt
from statsmodels.tsa.api import ExponentialSmoothing 

df = pd.read_csv("BUNDESBANK-BBK01_WT5511.csv")
df.head()

y = df['Value']
y_train = df['Value'][:-12]
y_test = df['Value'][-12:]

plt.plot(y_train, label='Train',color='blue')
plt.plot(y_test, label='Test',color='orange')
plt.legend(loc='best')
plt.show()

alpha = 0.3
beta = 0.02
gamma = 0.8
hw_add = ExponentialSmoothing(y_train, seasonal_periods=12, 
                            trend='add', seasonal='add')
fit1 = hw_add.fit(smoothing_level=alpha, 
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(100,750, "RMSE="+str(error))
params_str = "Alpha="+str(fit1.params['smoothing_level'])
params_str = params_str+"\n Beta="+str(fit1.params['smoothing_trend'])
params_str = params_str+"\n Gamma="+str(fit1.params['smoothing_seasonal'])
plt.text(100,1000,params_str )
plt.title("HW Additive Trend and Seasonal Method")
plt.legend(loc='best')
plt.show()

### Multiplicative
alpha = 0.3
beta = 0.02
gamma = 0.8
hw_mul = ExponentialSmoothing(y_train, seasonal_periods=12, 
                            trend='add', seasonal='mul')
fit1 = hw_mul.fit(smoothing_level=alpha, 
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(100,750, "RMSE="+str(error))
params_str = "Alpha="+str(fit1.params['smoothing_level'])
params_str = params_str+"\n Beta="+str(fit1.params['smoothing_trend'])
params_str = params_str+"\n Gamma="+str(fit1.params['smoothing_seasonal'])
plt.text(100,1000,params_str )
plt.title("HW Additive Trend and Multiplicative Seasonal Method")
plt.legend(loc='best')
plt.show()

#### Damping

alpha = 0.3
beta = 0.02
gamma = 0.8
phi = 0.4
hw_add = ExponentialSmoothing(y_train, seasonal_periods=12, 
                            trend='add', seasonal='add', 
                            damped_trend=True)
fit1 = hw_add.fit(smoothing_level=alpha, 
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma,
                    damping_trend=phi)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(100,750, "RMSE="+str(error))
params_str = "Alpha="+str(fit1.params['smoothing_level'])
params_str = params_str+"\n Beta="+str(fit1.params['smoothing_trend'])
params_str = params_str+"\n Gamma="+str(fit1.params['smoothing_seasonal'])
params_str = params_str+"\n Phi="+str(fit1.params['damping_trend'])
plt.text(100,1000,params_str )
plt.title("HW Additive Trend and Seasonal Method")
plt.legend(loc='best')
plt.show()

### Multiplicative
alpha = 0.3
beta = 0.02
gamma = 0.8
hw_mul = ExponentialSmoothing(y_train, seasonal_periods=12, 
                            trend='add', seasonal='mul',
                            damped_trend=True)
fit1 = hw_mul.fit(smoothing_level=alpha, 
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma,
                    damping_trend=phi)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(100,750, "RMSE="+str(error))
params_str = "Alpha="+str(fit1.params['smoothing_level'])
params_str = params_str+"\n Beta="+str(fit1.params['smoothing_trend'])
params_str = params_str+"\n Gamma="+str(fit1.params['smoothing_seasonal'])
params_str = params_str+"\n Phi="+str(fit1.params['damping_trend'])
plt.text(100,1000,params_str )
plt.title("HW Additive Trend and Multiplicative Seasonal Method")
plt.legend(loc='best')
plt.show()

