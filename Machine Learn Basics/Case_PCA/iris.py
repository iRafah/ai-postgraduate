from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Target'] = iris.get('target')
print(df.head())

features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
X = df[features].values
y = df['Target'].values

# Normalizing the data using standardScaler
# (Standardizes the features by removing the mean and scales the variance to one unit.
# This means that for each feature, the mean would be 0, and the Standard Deviation would be 1)
X = StandardScaler().fit_transform(X)
df_standardized = pd.DataFrame(data=X, columns=features)
print(df_standardized.head())

pca = PCA(n_components=2)
mainComponents = pca.fit_transform(X)

df_pca = pd.DataFrame(data=mainComponents, columns=['PC1', 'PC2'])
target = pd.Series(iris['target'], name='target')
result_df = pd.concat([df_pca, target], axis=1)
print(result_df)


print('Variance of each component:', pca.explained_variance_ratio_)
print('\n Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_))*100, 2))

results = []
X = df_standardized
for n in range(2, 5):
    pca = PCA(n_components=n)

    pca.fit(X)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    results.append(explained_variance)

plt.figure(figsize=(15, 8))
plt.plot(range(2, 5), results, marker='o', linestyle='-', color='b')
plt.xlabel('Number of components')
plt.ylabel('Cumulative Variance Explained (%)')
plt.title('Cumulative Variance Explained by components of PCA')
plt.grid(True)

for i, (n_components, explained_var) in enumerate(zip(range(2, 5), results)):
    plt.text(n_components, explained_var, f'Component {n_components}', ha='right', va='bottom')
plt.show()
