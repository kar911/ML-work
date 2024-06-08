import os
os.chdir("C:/Training/Academy/Statistics (Python)/Datasets")

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_squared_error
from numpy import sqrt

df = pd.read_csv("monthly-milk-production-pounds-p.csv")
df.head()

df.plot.line(x = 'Month',y = 'Milk')
plt.show()

y = df['Milk']
#### Centered MA
fcast = y.rolling(3,center=True).mean()
plt.plot(y, label='Data')
plt.plot(fcast, label='Centered Moving Average')
plt.legend(loc='best')
plt.show()



y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]
span = 5
#### Trailing MA
fcast = y_train.rolling(span).mean()
MA = fcast.iloc[-1]
MA_series = pd.Series(MA.repeat(len(y_test)))
MA_fcast = pd.concat([fcast,MA_series],ignore_index=True)
plt.plot(y_train, label='Train')
plt.plot(y_test, label='Test')
plt.plot(MA_fcast, label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(y_test, MA_series))
print(rms)
