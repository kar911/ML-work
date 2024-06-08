import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

groceries = []
with open("groceries.csv","r") as f:groceries = f.read()
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
  
te = TransactionEncoder()
te_ary = te.fit(groceries_list).transform(groceries_list)
te_ary

fp_df = pd.DataFrame(te_ary, columns=te.columns_)

##### DatasetA

groceries = []
with open("DatasetA.csv","r") as f:groceries = f.read()
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
  
te = TransactionEncoder()
te_ary = te.fit(groceries_list).transform(groceries_list)
te_ary

fp_df = pd.DataFrame(te_ary, columns=te.columns_)
fp_df = fp_df.iloc[:,1:]
