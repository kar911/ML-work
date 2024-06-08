import pandas as pd 
import os 
os.chdir(r"C:\Training\Kaggle\Competitions\What is Cooking")
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

train = pd.read_json('train.json')

ingred_list = train['ingredients'].tolist()

te = TransactionEncoder()
te_ary = te.fit(ingred_list).transform(ingred_list)
fp_df = pd.DataFrame(te_ary, columns=te.columns_)

# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.01,
                   use_colnames=True)
# and convert into rules
rules = association_rules(itemsets,
                          metric='confidence', 
                          min_threshold=0.6)
rules = rules[['antecedents', 'consequents',
               'support', 'confidence', 'lift']]
rules = rules.sort_values('lift', ascending=False)

################ Classifier
rf = RandomForestClassifier(random_state=23,n_estimators=50,
                            max_depth=7)
kfold = StratifiedKFold(n_splits=5, 
                        shuffle=True, random_state=23)
params = {'max_features':[100, 150]}
y = train['cuisine']
gcv = GridSearchCV(rf, param_grid=params,
                   cv=kfold, verbose=3)
gcv.fit(fp_df,y)
print(gcv.best_params_)
print(gcv.best_score_)

best = gcv.best_estimator_

importances = best.feature_importances_
pd_imp = pd.DataFrame({'Feature':list(fp_df.columns),
                       'Importance':importances})
pd_imp.sort_values(by='Importance',ascending=False,
                                inplace=True)
top_imp_10 = pd_imp.iloc[:10,:]
plt.barh(top_imp_10['Feature'], top_imp_10['Importance'])
plt.title("Feature Importances Plot")
plt.show()

