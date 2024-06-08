import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error as mse
from statsmodels.graphics.tsaplots import plot_acf
import numpy as np 

df = pd.read_csv("monthly-milk-production-pounds-p.csv")
plot_acf(df['Milk'], lags=7)
plt.show()

wgem = pd.read_csv("WGEM-IND_CPTOTNSXN.csv")
plot_acf(wgem['Value'], lags=7)
plt.show()


y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]
###### ARMA Models #############
from statsmodels.tsa.arima.model import ARIMA
# train MA
model = ARIMA(y_train,order=(5,0,1))
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
# make predictions
fcast1 = model_fit.predict(start=len(y_train), 
                           end=len(y_train)+len(y_test)-1, 
                           dynamic=False)
error = round(sqrt(mse(y_test, fcast1)),2)
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600, "RMSE="+str(error))
plt.legend(loc='best')
plt.show()



# train MA
model = ARIMA(y_train,order=(5,2,1))
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
# make predictions
fcast1 = model_fit.predict(start=len(y_train), 
                           end=len(y_train)+len(y_test)-1, 
                           dynamic=False)
error = round(sqrt(mse(y_test, fcast1)),2)
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.text(100,600, "RMSE="+str(error))
plt.legend(loc='best')
plt.show()


############# A-Dicky Fuller Test ########
from statsmodels.tsa.stattools import adfuller
y= df['Milk']
ad_result = adfuller(y, autolag=None)

p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")

## 1st order diff
d_y = y - y.shift(1)
ad_result = adfuller(d_y.iloc[1:], autolag=None)
p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")

y= wgem['Value']
ad_result = adfuller(y, autolag=None)

p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")

## 1st order diff
d_y = y - y.shift(1)
ad_result = adfuller(d_y.iloc[1:], autolag=None)
p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")
    
## 2nd order diff
d_d_y = d_y - d_y.shift(1)
ad_result = adfuller(d_d_y.iloc[2:], autolag=None)
p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")

######################################
gas = pd.read_csv("Gasoline.csv")    
y = gas['Sales']  
ad_result = adfuller(y, autolag=None)

p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")
    
## 1st order diff
d_y = y - y.shift(1)
ad_result = adfuller(d_y.iloc[1:], autolag=None)
p_val = ad_result[1]
if p_val < 0.05:
    print("Stationary")
else:
    print("Non-Stationary")
    