import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

housing = pd.read_csv("../Datasets/Housing.csv")
intervals = [(0, 50000), (50000, 100000), (100000, 150000),
             (150000, 200000)]
bins = pd.IntervalIndex.from_tuples(intervals)
housing['price_slab'] = pd.cut(housing['price'], bins)

intervals = [(0, 5000), (5000, 10000), (10000, 15000),
             (15000, 20000)]
bins = pd.IntervalIndex.from_tuples(intervals)
housing['area_slab'] = pd.cut(housing['lotsize'], bins)

housing.drop(['price','lotsize'], axis=1, inplace=True)

housing = housing.astype(object)
fp_df = pd.get_dummies(housing, prefix_sep='=')

# create frequent itemsets
itemsets = apriori(fp_df, min_support=0.2, use_colnames=True)
# and convert into rules
rules = association_rules(itemsets, 
                          metric='confidence', min_threshold=0.6)
rules = rules[['antecedents', 'consequents', 
               'support', 'confidence', 'lift']]
rules = rules[rules['lift']>1]
rules = rules.sort_values(by='lift', ascending=False)
print(rules)
