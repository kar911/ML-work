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
###### AutoRegressive Models #############
from statsmodels.tsa.arima.model import ARIMA
# train MA
model = ARIMA(y_train,order=(5,0,0))
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

scores = []
ar = [1,2,3,4,5,6]
for a in ar:
    model = ARIMA(y_train,order=(a,0,0))
    model_fit = model.fit()
    fcast1 = model_fit.predict(start=len(y_train), 
                               end=len(y_train)+len(y_test)-1, 
                               dynamic=False)
    scores.append(round(sqrt(mse(y_test, fcast1)),2))

i_min = np.argmin(scores) 
print("Best AR order =", ar[i_min])
print("Best Score =", scores[i_min])
