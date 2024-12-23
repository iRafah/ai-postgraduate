
import pandas as pd                      
import matplotlib.pyplot as plt          
import seaborn as sns                    
import numpy as np                        

from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree  


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

# Getting the data
dados = pd.read_csv('card_transdata.csv', sep=',')

print(dados.head())

print(dados.isnull().sum())

dados = dados.dropna()
print(dados.isnull().sum())

correction_matrix = dados.corr().round(2)
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(data=correction_matrix, annot=True, linewidths=.5, ax=ax)

# plt.show()

# Splitting the data
x = dados.drop(columns=['fraud'])
y = dados['fraud']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=7) 

# Creating the tree model
# Decision tree
dt = DecisionTreeClassifier(random_state=7, criterion='entropy', max_depth=2)
dt.fit(x_train, y_train)

y_predito = dt.predict(x_test)
tree.plot_tree(dt)

# plt.show()

class_names = ['Fraude', 'Não Fraude']
label_names = ['distance_from_home', 'distance_from_last_transaction',	'ratio_to_median_purchase_price',	'repeat_retailer',	'used_chip',	'used_pin_number',	'online_order']

fig, axes = plt.subplots(nrows = 1, ncols=1, figsize=(15, 15), dpi=300)
tree.plot_tree(dt,
               feature_names=label_names,
               class_names=class_names,
               filled=True)

fig.savefig('treegraph.png')

print(accuracy_score(y_test, y_predito))

# Random forest
rf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=7)
rf.fit(x_train, y_train)

estimator = rf.estimators_
y_predito_random_forest = rf.predict(x_test)

# Precision, recall, f1-score and accuracy metrics.
print(accuracy_score(y_test, y_predito_random_forest))

class_names = ['Fraude', 'Não Fraude']
label_names = ['distance_from_home', 'distance_from_last_transaction',	'ratio_to_median_purchase_price',	'repeat_retailer',	'used_chip',	'used_pin_number',	'online_order']

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=800)
tree.plot_tree(rf.estimators_[0],
               feature_names=label_names,
               class_names=class_names,
               filled=True)
fig.savefig('rf_individualTree.png')

# Plotting all the trees
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 2), dpi=900)
for index in range(0, 5):
    tree.plot_tree(rf.estimators_[index],
                   feature_names=label_names,
                   class_names=class_names,
                   filled=True,
                   ax=axes[index]);
    axes[index].set_title('Estimator:' + str(index), fontsize=11)

fig.savefig('rf_5Trees.png')


print (rf.score(x_train, y_train)) 
print(rf.score(x_test, y_test))