import pandas as pd 
import numpy as np 

train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")

print(np.sum(train['Item_Weight'].isnull()))
print(train['Item_Fat_Content'].value_counts())

train['Item_Fat_Content'].replace(to_replace="reg", 
                                  value="Regular",
                                  inplace=True)
train['Item_Fat_Content'].replace(to_replace="LF", 
                                  value="Low Fat",
                                  inplace=True)
train['Item_Fat_Content'].replace(to_replace="low fat", 
                                  value="Low Fat",
                                  inplace=True)

weights = train.groupby('Item_Identifier')['Item_Weight'].mean()
weights = weights.reset_index()
weights.rename(columns={'Item_Weight':'i_weigh'},
               inplace=True)
train = train.merge(weights, on='Item_Identifier')
#train[['Item_Identifier','Item_Weight','i_weigh']]

outlets = train[['Outlet_Identifier',
'Outlet_Establishment_Year', 'Outlet_Size', 
'Outlet_Location_Type',
'Outlet_Type']].drop_duplicates()

train['Outlet_Size'].fillna('Small', 
                              inplace=True)

train.drop('Item_Weight', axis=1, inplace=True)
train.rename(columns={'i_weigh':'Item_Weight'},
               inplace=True)