import os
os.chdir(r"C:\Training\Kaggle\Competitions\Playground Competitions\Horse Survival")
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder 

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
ss = pd.read_csv("sample_submission.csv")

## columns having at least one null value in train
null_cols_trn = list(train.columns[pd.isnull(train).sum()>0])

## columns having at least one null value in test
null_cols_tst = list(test.columns[pd.isnull(test).sum()>0])

imp = SimpleImputer(strategy="constant", 
 fill_value="missing").set_output(transform="pandas")

X_train = train.drop('outcome', axis=1)

imp_trn = imp.fit_transform(X_train)
## columns having at least one null value in train
null_cols_trn = list(imp_trn.columns[pd.isnull(imp_trn).sum()>0])
print(null_cols_trn)

imp_tst = imp.transform(test)

## columns having at least one null value in test
null_cols_tst = list(imp_tst.columns[pd.isnull(imp_tst).sum()>0])

##### One hot encoding
list_obj = ['surgery','age', 
            'temp_of_extremities','peripheral_pulse']
ohc = OneHotEncoder(sparse_output=False, 
                    categories=).set_output(transform="pandas")

ohc_imp_trn = ohc.fit_transform(imp_trn)








