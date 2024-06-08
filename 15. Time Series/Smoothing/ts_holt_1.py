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

# Holt's Linear Method
alpha = 0.7
beta = 0.3
### Linear Trend
from statsmodels.tsa.api import Holt
holt = Holt(y_train)
fit1 = holt.fit(smoothing_level=alpha,
                smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(5,110, "RMSE="+str(error))
plt.title("Holt's Linear Trend")
plt.legend(loc='best')
plt.show()

print(fit1.params)

#### FMAC
fmac = pd.read_csv("FMAC-HPI_24420.csv")
y = fmac['NSA Value']
y_train = y.iloc[:-8]
y_test = y.iloc[-8:]

alpha = 0.02
beta = 0.22
### Linear Trend
from statsmodels.tsa.api import Holt
holt = Holt(y_train)
fit1 = holt.fit(smoothing_level=alpha,
                smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(5,110, "RMSE="+str(error))
plt.title("Holt's Linear Trend")
plt.legend(loc='best')
plt.show()

print(fit1.params)


scores = []
for alpha in np.linspace(0, 1, 100):
    for beta in np.linspace(0, 1, 100):
        holt = Holt(y_train)
        fit1 = holt.fit(smoothing_level=alpha,
                        smoothing_trend=beta)
        fcast1 = fit1.forecast(len(y_test))
        sc = sqrt(mse(y_test, fcast1))
        scores.append([alpha, beta, sc])
        
pd_scores = pd.DataFrame(scores,
                         columns=['alpha','beta','score'])
pd_scores.sort_values('score')

### Expoential Trend
alpha = 0.02
beta = 0.22
from statsmodels.tsa.api import Holt
holt = Holt(y_train,  exponential=True)
fit1 = holt.fit(smoothing_level=alpha,
                smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(5,110, "RMSE="+str(error))
plt.title("Holt's Exp Trend")
plt.legend(loc='best')
plt.show()

print(fit1.params)


scores = []
for alpha in np.linspace(0, 1, 100):
    for beta in np.linspace(0, 1, 100):
        holt = Holt(y_train, exponential=True)
        fit1 = holt.fit(smoothing_level=alpha,
                        smoothing_trend=beta)
        fcast1 = fit1.forecast(len(y_test))
        sc = sqrt(mse(y_test, fcast1))
        scores.append([alpha, beta, sc])
        
pd_scores = pd.DataFrame(scores,
            columns=['alpha','beta','score'])
pd_scores.sort_values('score')

### Addtitive
alpha = 0.02
beta = 0.22
phi = 0.4
holt = Holt(y_train,  damped_trend=True)
fit1 = holt.fit(smoothing_level=alpha,
                smoothing_trend=beta,
                damping_trend=phi)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(5,110, "RMSE="+str(error))
plt.title("Holt's Add Damped Trend")
plt.legend(loc='best')
plt.show()

print(fit1.params)

