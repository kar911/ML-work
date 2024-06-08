import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer 

a = np.array([[23,	4,	0.5],
               [45,	np.nan	,0.38],
	          [np.nan, 6,	0.4],
              [56,	7,np.nan],	
              [66,	2,	0.32]])
pa = pd.DataFrame(a, columns=['x1','x2','x3'])

pa.mean()
pa.fillna({'x1':47.5, 'x2':4.75, 'x3': 0.4})
# OR

imp = SimpleImputer(strategy='median').set_output(transform="pandas")
imp.fit_transform(pa)
