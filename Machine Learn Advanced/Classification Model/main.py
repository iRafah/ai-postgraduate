import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_excel('gaf_esp.xlsx')
data.head()

data.describe()
data.groupby('Espécie').describe()

data.plot.scatter(x='Comprimento do Abdômen', y='Comprimento das Antenas')

# plt.show()

x = data[['Comprimento do Abdômen', 'Comprimento das Antenas']]
y = data['Espécie']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

print(list(y_train).count('Gafanhoto'))

print(list(y_train).count('Esperança'))

print("Total base de treino: ", len(x_train))
print("Total base de teste: ", len(y_test))

# Hiperparametro do nosso modelo é o número de vizinhos considerado (n_neighbors)
modelo_classificador = KNeighborsClassifier(n_neighbors=3)

# Está fazendo o treinamento do meu modelo de ML
modelo_classificador.fit(x_train, y_train)

modelo_classificador.predict([[8, 6]]) # Esperança

y_predito = modelo_classificador.predict(x_test)

result = accuracy_score(y_true = y_test, y_pred=y_predito)
print(result)
