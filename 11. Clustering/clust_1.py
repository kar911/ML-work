import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel("Clust_example.xlsx")
df.set_index('Obs', inplace=True)

plt.scatter(df['X1'], df['X2'])
plt.show()
scaler = MinMaxScaler().set_output(transform='pandas')
df_scaled = scaler.fit_transform(df)
mergings = linkage(df,
                   method='complete')
dendrogram(mergings,labels=list(df.index))
plt.show()

##############################
milk = pd.read_csv("milk.csv",index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
milkscaled=scaler.fit_transform(milk)

# Calculate the linkage: mergings
link= 'single'
mergings = linkage(milkscaled,method=link)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=np.array(milk.index),
           leaf_rotation=45,
           leaf_font_size=10)
plt.title(link+" linkage")
plt.show()
