import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Feature Engineer

# Plot dos gráficos
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

# Algorítmos de agrupamento
from sklearn.cluster import KMeans, DBSCAN

# Avaliação de desempenho
from sklearn.metrics import adjusted_rand_score, silhouette_score

dados = pd.read_csv('mall.csv', sep=',')

print(dados.shape)
print(dados.head())

print(dados.isnull().sum())

# Análise exploratória dos dados
print(dados.describe())

print(dados['Annual Income (k$)'].median())

# dados.hist(figsize=(12, 12)) # histograma

plt.figure(figsize=(6,4))
# sns.heatmap(dados[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr(method='pearson'), annot=True, fmt='.1f')
# plt.show()

print(dados['Gender'].value_counts())
# sns.pairplot(dados, hue='Gender')
# plt.show()

scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(dados[['Annual Income (k$)', 'Spending Score (1-100)']])
dados_escalonados = scaler.transform(dados[['Annual Income (k$)', 'Spending Score (1-100)']])

print(dados_escalonados)

# Sem feature scaler
# Definindo o module de clusterização. K-MEANS com 6 clusters.
kmeans = KMeans (n_clusters=6, random_state=0)

kmeans.fit(dados[['Annual Income (k$)', 'Spending Score (1-100)']])
centroides = kmeans.cluster_centers_

kmeans_labels = kmeans.predict(dados[['Annual Income (k$)', 'Spending Score (1-100)']])

# Com feature scaler
# Definindo o module de clusterização. K-MEANS com 6 clusters.
kmeans_escalonados = KMeans(n_clusters=6, random_state=0)

kmeans.fit(dados_escalonados)
centroides_escalonados = kmeans.cluster_centers_

kmeans_labels_escalonado = kmeans.predict(dados_escalonados)

dados_escalonados = pd.DataFrame(dados_escalonados, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
print(dados_escalonados.head())

dados_escalonados['Grupos'] = kmeans_labels_escalonado
print(dados_escalonados.head())

dados['Grupos'] = kmeans_labels
print(dados.head())

# Vamos analisar a nosso previsão e os centroides
print(pd.Series(kmeans_labels).value_counts())

print(centroides)

# Plotando os dados identificando com os seus clusters
plt.scatter(dados_escalonados[['Annual Income (k$)']], dados_escalonados[['Spending Score (1-100)']], c=kmeans_labels_escalonado, s=200, alpha=0.5, cmap='rainbow')
plt.xlabel('Salário Anual')
plt.ylabel('Pontuação de gastos')

plt.scatter(centroides_escalonados[:, 0], centroides_escalonados[:, 1],  c='black', marker='X', s=200, alpha=0.5)
plt.rcParams['figure.figsize'] = (10, 5)
# plt.show()

# Plotando os dados identificando com os seus clusters
plt.scatter(dados[['Annual Income (k$)']], dados[['Spending Score (1-100)']], c=kmeans_labels, alpha=0.5, cmap='rainbow')
plt.xlabel('Salário Anual')
plt.ylabel('Pontuação de gastos')

plt.scatter(centroides[:, 0], centroides[:, 1],  c='black', marker='X', s=200, alpha=0.5)
plt.rcParams['figure.figsize'] = (10, 5)
# plt.show()

# Vamos tentar descobrir qual é o melhor K.
# Lista com a quantidade de clusters que iremos testar
k = list(range(1, 10))

# Armazena o SSE (soma dos erros quadráticos) para cada quantidade de k
sse = []

# Roda o K-means para cada K

for i in k:
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(dados[['Annual Income (k$)', 'Spending Score (1-100)']])
    sse.append(kmeans.inertia_) # cálculo do erro do k-means

plt.rcParams['figure.figsize'] = (10, 5)
plt.plot(k, sse, '-o')
plt.xlabel('Número de clusters')
plt.ylabel('Inércia')
# plt.show()

print(dados.groupby('Grupos')['Age'].mean())
print(dados.groupby('Grupos')['Annual Income (k$)'].mean())

# Definindo o modelo de clusterizacao. K-MEANS com 5 clusters
kmeans = KMeans(n_clusters=5,random_state=0)

#Implementando o K-Means nos dados:
kmeans.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])

#Salvando os centroides de cada cluster
centroides = kmeans.cluster_centers_

#Salvando os labels dos clusters para cada exemplo
kmeans_labels = kmeans.predict(dados[['Annual Income (k$)','Spending Score (1-100)']])

# plotando os dados identificando com os seus clusters
plt.scatter(dados[['Annual Income (k$)']],dados[['Spending Score (1-100)']], c=kmeans_labels, alpha=0.5, cmap='rainbow')
plt.xlabel('Salario Anual')
plt.ylabel('Pontuação de gastos')
# plotando os centroides
plt.scatter(centroides[:, 0], centroides[:, 1], c='black', marker='X', s=200, alpha=0.5)
plt.rcParams['figure.figsize'] = (10, 5)
# plt.show()

# Visualizar alguns dados
dados_grupo_1 = dados[dados['Grupos'] == 1]
dados_grupo_2 = dados[dados['Grupos'] == 2]
dados_grupo_3 = dados[dados['Grupos'] == 3]
dados_grupo_4 = dados[dados['Grupos'] == 4]
dados_grupo_1['Annual Income (k$)'].mean() #grupo 1 azul
dados_grupo_2['Annual Income (k$)'].mean() #grupo 2 roxo
dados_grupo_3['Annual Income (k$)'].mean() #grupo 3 laranja
dados_grupo_3['Age'].mean()
dados_grupo_2['Age'].mean() #grupo 2 roxo
dados_grupo_4['Annual Income (k$)'].mean() #grupo 4 vermelho
dados_grupo_3['Spending Score (1-100)'].mean() #grupo 4

############################################################
# DBSCAN

dbscan = DBSCAN(eps=10, min_samples=8)
# Ajustando aos dados
dbscan.fit(dados[['Annual Income (k$)', 'Spending Score (1-100)']])

dbscan_labels = dbscan.labels_
print(dbscan_labels)

plt.scatter(dados[['Annual Income (k$)']], dados[['Spending Score (1-100)']], c=dbscan_labels, alpha=0.5, cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
# plt.show()

# Plotando o gráfico sem os outliers
# máscara para outlier
mascara = dbscan_labels>=0

# Plotando o gráfico
plt.scatter(dados[['Annual Income (k$)']][mascara], dados[['Spending Score (1-100)']][mascara], c=dbscan_labels[mascara], alpha=0.5, cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
# plt.show()

# Valores que foram considerados outliers
print(list(mascara).count(False))

print('Comparação entre K-Means e DBSCAN')
print('TIPO EXTERNO:')
print(adjusted_rand_score(kmeans_labels, dbscan_labels))

print('TIPO INTERNO:')
print('KMEANS:')
print(silhouette_score(dados[['Annual Income (k$)', 'Spending Score (1-100)']], kmeans_labels))

print('DBSCAN:')
print(silhouette_score(dados[['Annual Income (k$)', 'Spending Score (1-100)']], dbscan_labels))
