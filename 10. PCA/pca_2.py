import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
import seaborn as sns
import matplotlib.pyplot as plt 

iris = pd.read_csv("iris.csv")

# xvar = 'Petal.Length'
# yvar = 'Petal.Width'

# sns.scatterplot(data=iris,
#                 x=xvar, y=yvar,
#                 hue='Species')
# plt.legend(loc='best')
# plt.show()

scaler = StandardScaler().set_output(transform='pandas')
i_scaled = scaler.fit_transform(iris.iloc[:,:-1])

prcomp = PCA().set_output(transform='pandas')
components = prcomp.fit_transform(i_scaled)
components['Species'] = iris['Species']

sns.scatterplot(data=components,
                x='pca0', y='pca1',
                hue='Species')
plt.legend(loc='best')
plt.show()

###############################################
from sklearn.preprocessing import LabelEncoder
glass = pd.read_csv("Glass.csv")

scaler = StandardScaler().set_output(transform='pandas')
i_scaled = scaler.fit_transform(glass.iloc[:,:-1])

prcomp = PCA().set_output(transform='pandas')
components = prcomp.fit_transform(i_scaled)
le = LabelEncoder()
components['Type'] = le.fit_transform(glass['Type'])
components['Type'] = components['Type'].astype(str)

sns.scatterplot(data=components,
                x='pca0', y='pca1',
                hue='Type')
plt.legend(loc='best')
plt.show()