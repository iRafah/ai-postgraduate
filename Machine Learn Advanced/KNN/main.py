import pandas as pd
import missingno as msno
import seaborn as sb
import matplotlib.pyplot as plt
import plotly_express as px
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

dados = pd.read_excel('Recrutamento.xlsx')
# print(dados.head()

print(dados.describe())
print(dados.shape)

# Inferência sobre os dados:
# Métricas de pontuação sobre ensino: ssc_p hsc_p degree_p estet_p mba_p
# sl_no é um código, então não faz sentido na análise.

# Visualizar dados nulos
# msno.matrix(dados)

print(dados.isnull().sum())

sb.boxplot(x='status', y='salary', data=dados, palette='hls', hue='status')

dados['salary'].fillna(value=0, inplace=True)
print(dados.isnull().sum())

# Vamos analisar os dados em dois gráficos diferentes
# boxplot e histplot
hsc_p_box = sb.boxplot(x=dados['hsc_p'])
hsc_p_hist = sb.histplot(data=dados, x='hsc_p')

sb.boxplot(x=dados["degree_p"])
sb.histplot(data=dados, x="degree_p")

sb.boxplot(x=dados["etest_p"])
sb.histplot(data=dados, x="etest_p")

sb.boxplot(x=dados["mba_p"])
sb.histplot(data=dados, x="mba_p")

sb.boxplot(x=dados["salary"])
sb.histplot(data=dados, x="salary")

# Gráfico que mostra se as pessoas com MBA foram contratados ou não e qual a ligação entre esses eixos
sb.set_theme(style='whitegrid', palette='muted')
ax=sb.swarmplot(data=dados, x='mba_p', y='status', hue='workex')
ax.set(ylabel='mba_p')
ax.set(xlabel='status')

violin_graph = px.violin(dados, y='salary', x='specialisation', color='gender', box=True, points='all')

# violin_graph.show()

# correlation_matrix = dados.corr().round(2)
# fig, ax = plt.subplots(figsize=(8,8))
# sb.heatmap(data=correlation_matrix, annot=True, linewidths=5, ax=ax)

# Há algumas colunas importantes que são do tipo texto. Vamos transformá-las em numerais.
# LabelEncoder funciona muito bem com textos que oferecem apenas 2 opcões (Yes - No, M - F, etc.)
print(dados.head())

colunas=['gender', 'workex', 'specialisation', 'status']

label_encoder = LabelEncoder()
for col in colunas:
    dados[col] = label_encoder.fit_transform(dados[col])
print(dados.head())

dummy_hsc_s = pd.get_dummies(dados['hsc_s'], prefix='dummy')
dummy_degree_t = pd.get_dummies(dados['degree_t'], prefix='dummy')

dados_dummy = pd.concat([dados, dummy_hsc_s, dummy_degree_t], axis=1)
print(dados_dummy)

# Vamos dropar as colunas que não vamos usar.
dados_dummy.drop(['hsc_s', 'degree_t', 'salary', 'hsc_b', 'ssc_b'], axis=1, inplace=True)
print(dados_dummy.head())

correlation_matrix = dados_dummy.corr().round(2)
fig, ax = plt.subplots(figsize=(14,14))
# sb.heatmap(data=correlation_matrix, annot=True, linewidths=5, ax=ax)

# A partir daqui vamos começar a criar o modelo preditivo.
x = dados_dummy[['ssc_p', 'hsc_p', 'degree_p', 'workex', 'mba_p']] # variáveis independentes
y = dados_dummy['status'] # variável alvo

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=7)

print(x_train.shape)
print(y_test.shape)

# Vamos padronizar os dados.
scaler = StandardScaler()
scaler.fit(x_train)

x_train_escalonado = scaler.transform(x_train)
x_test_escalonado = scaler.transform(x_test)

print(x_train_escalonado)

error = []

for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)    
    knn.fit(x_train_escalonado, y_train)
    pred_i = knn.predict(x_test_escalonado)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize='10')

plt.title('Erro médio para K')
plt.xlabel('Valor de K')
plt.ylabel('Erro médio')
# plt.show() # show gráfico

modelo_classificador = KNeighborsClassifier(n_neighbors=5)
modelo_classificador.fit(x_train_escalonado, y_train)

y_predito = modelo_classificador.predict(x_test_escalonado)
print(y_predito)

print(accuracy_score(y_test, y_predito))

svm = Pipeline(
    [
        ('linear_svc', LinearSVC(C=1))
    ]
)
svm.fit(x_train_escalonado, y_train)

y_predito_svm = svm.predict(x_test_escalonado)
print(accuracy_score(y_test, y_predito_svm))