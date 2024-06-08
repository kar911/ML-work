import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import KFold, GridSearchCV

x = np.array([[2,4],
              [3,5],
              [12,18],
              [15,20],
              [34,56],
              [35,60],
              [78, 26],
              [80, 23]])
y = np.array([100, 123, 45, 67, 230, 245, 34, 20])

p_df = pd.DataFrame(x, columns=['x1', 'x2'])
p_df['y'] = y

sns.scatterplot(data=p_df, x='x1',y='x2',
                hue='y')
plt.show()


X_train = p_df[['x1','x2']]
y_train = p_df['y']

dtr = DecisionTreeRegressor(random_state=23)
dtr.fit(X_train, y_train)

plt.figure(figsize=(25,10))
plot_tree(dtr,feature_names=list(X_train.columns),
               filled=True,fontsize=18)
plt.show() 

###############################################
housing = pd.read_csv("Housing.csv")
dum_hous = pd.get_dummies(housing, drop_first=True)
X = dum_hous.drop('price', axis=1)
y = dum_hous['price']

dtr = DecisionTreeRegressor(random_state=23)
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=23)
params = {'max_depth':[2,3,4,5,6,7,None],
          'min_samples_split':[2, 5, 10, 20],
          'min_samples_leaf':[1,5,7,10,20]}
gcv_tree = GridSearchCV(dtr, param_grid=params,
                        cv=kfold, verbose=3)
gcv_tree.fit(X, y)
print(gcv_tree.best_params_)
print(gcv_tree.best_score_)
pd_cv = pd.DataFrame(gcv_tree.cv_results_)

#### Best Tree
best_tree = gcv_tree.best_estimator_
plt.figure(figsize=(35,15))
plot_tree(best_tree,feature_names=list(X.columns),
               filled=True,fontsize=12)
plt.show() 


importances = best_tree.feature_importances_
pd_imp = pd.DataFrame({'Feature':list(X.columns),
                       'Importance':importances})
pd_imp.sort_values(by='Importance', inplace=True)
plt.barh(pd_imp['Feature'], pd_imp['Importance'])
plt.title("Feature Importances Plot")
plt.show()
