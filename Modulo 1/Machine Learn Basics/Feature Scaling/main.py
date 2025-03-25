import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Churn_Modelling.csv', sep=';')
df.head()

df.shape

# Criar o gráfico de boxplot
plt.boxplot(df['CreditScore'])
plt.title('CreditScore')
plt.ylabel('Valores')
# plt.show()

print(df['CreditScore'].min())
print(df['CreditScore'].max())

# Criar o gráfico de boxplot
plt.boxplot(df['Age'])
plt.title('Age')
plt.ylabel('Valores')
# plt.show()

print(df['Age'].min())
print(df['Age'].max())

# Criar o gráfico de boxplot
plt.boxplot(df['Tenure'])
plt.title('Tenure')
plt.ylabel('Valores')
# plt.show()

print(df['Tenure'].min())
print(df['Tenure'].max())

# Criar o gráfico de boxplot
plt.boxplot(df['Balance'])
plt.title('Balance')
plt.ylabel('Valores')
# plt.show()


print(df['Balance'].min())
print(df['Balance'].max())

# Criar o gráfico de boxplot
plt.boxplot(df['NumOfProducts'])
plt.title('NumOfProducts')
plt.ylabel('Valores')
# plt.show()

print(df['NumOfProducts'].min())
print(df['NumOfProducts'].max())

# Criar o gráfico de boxplot
plt.boxplot(df['EstimatedSalary'])
plt.title('EstimatedSalary')
plt.ylabel('Valores')
# plt.show()

print(df['EstimatedSalary'].min())
print(df['EstimatedSalary'].max())

print(df.head())

label_encoder = LabelEncoder()

# Ajustar e transformar os rótulos
df['Surname'] = label_encoder.fit_transform(df['Surname'])
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])

print(df.head())

X = df.drop(columns=['Exited']) # Variáveis características
y = df['Exited'] # O que eu quero prever. (Target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler() # Chamando o método de normalização dos dados (0-1)
scaler.fit(X_train)

x_train_min_max_scaled = scaler.transform(X_train)
x_test_min_max_scaled = scaler.transform(X_test)

print(x_train_min_max_scaled)

scaler = StandardScaler() # Chamando o método de padronização dos dados (média e std)
scaler.fit(X_train) # Qual média e std será utilizado para o escalonamento

x_train_standard_scaled = scaler.transform(X_train)
x_test_standard_scaled = scaler.transform(X_test)

print(x_train_standard_scaled)

model = KNeighborsClassifier(n_neighbors=3)
# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)

print(f'Acurácia: {accuracy:.2f}')

# Testando com a normalização
model_min_max = KNeighborsClassifier(n_neighbors=3)
# Treinar o modelo
model_min_max.fit(x_train_min_max_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred_min_max = model.predict(x_test_min_max_scaled)

accuracy_min_max = accuracy_score(y_test, y_pred_min_max)
print(f'Acurácia: {accuracy_min_max:.2f}')

# Testando com a padronização
model_standard = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
model_standard.fit(x_train_standard_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred_standard = model.predict(x_test_standard_scaled)

accuracy_strandard = accuracy_score(y_test, y_pred_standard)
print(f'Acurácia: {accuracy_strandard:.2f}')