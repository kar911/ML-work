import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

cancer = pd.read_csv("Cancer.csv", index_col=0)
dum_canc = pd.get_dummies(cancer)

# create frequent itemsets
itemsets = apriori(dum_canc, min_support=0.2,
                   use_colnames=True)
# and convert into rules
rules = association_rules(itemsets,
                          metric='confidence', 
                          min_threshold=0.6)

rules = rules[['antecedents', 'consequents',
               'support', 'confidence', 'lift']]
rules = rules.sort_values('lift', ascending=False)

