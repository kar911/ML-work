import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

fp_df = pd.read_csv('Faceplate.csv',index_col=0)
fp_df = fp_df.astype(bool)
# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.2,
                   use_colnames=True)
# and convert into rules
rules = association_rules(itemsets,
                          metric='confidence', 
                          min_threshold=0.6)

rules = rules[['antecedents', 'consequents',
               'support', 'confidence', 'lift']]
rules.sort_values('lift', ascending=False)

#################################################

fp_df = pd.read_csv("Cosmetics.csv", index_col=0)
fp_df = fp_df.astype(bool)

# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.2,
                   use_colnames=True)
# and convert into rules
rules = association_rules(itemsets,
                          metric='confidence', 
                          min_threshold=0.6)

rules = rules[['antecedents', 'consequents',
               'support', 'confidence', 'lift']]
rules.sort_values('lift', ascending=False)

