import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("monthly-milk-production-pounds-p.csv")
df.head()

df.plot()
plt.show()

series = df['Milk']
result = seasonal_decompose(series, model='additive',period=12)
result.plot()
plt.title("Additive Decomposition")
plt.show()

result = seasonal_decompose(series, model='multiplicative',period=12)
result.plot()
plt.title("Multiplicative Decomposition")
plt.show()

############## Air Passengers #################
air = pd.read_csv("AirPassengers.csv")

air.plot()
plt.show()

series = air['Passengers']
result = seasonal_decompose(series, model='additive',period=12)
result.plot()
plt.title("Additive Decomposition")
plt.show()

result = seasonal_decompose(series, model='multiplicative',period=12)
result.plot()
plt.title("Multiplicative Decomposition")
plt.show()

