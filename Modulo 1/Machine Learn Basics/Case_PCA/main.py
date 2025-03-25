import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import shapiro

# Vamos carregar o arquivo csv em nosso drive e analisarmos o nosso dataframe
pd.set_option('display.max_columns', None)

df_fifa = pd.read_csv('players_22.csv')
# print(df_fifa)

print(df_fifa.shape)

# print(df_fifa.describe(0))

print("Informações sobre o DataFrame df_fifa:")
print(df_fifa.info())

# Criando dataframe somente com nossas variáveis numericas
df_fifa_numerico = df_fifa.select_dtypes([np.number])
# print(df_fifa_numerico)

# Calcula a matriz de correlação
correlation_matrix = df_fifa_numerico.corr()
print(correlation_matrix)

# Visualização da matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5)
plt.title('Matriz de correlação')
# plt.show()

# Vamos verificar quantidades de nulos em nosso dataframe, não podemos aplicar o PCA
# se nossos dados tiverem linhas nulas, devemos tratar esses casos caso ocorram
# Existem diversas técnicas para tratar dados nulos, o método de escolha depende muito do seu objetivo
print(df_fifa_numerico.isnull().sum())

# Preenche os valores NaN com a média das colunas
imputer = SimpleImputer(strategy='mean')
df_fifa_numerico = pd.DataFrame(imputer.fit_transform(df_fifa_numerico), columns=df_fifa_numerico.columns)

scaler = StandardScaler()
df_fifa_padronizado = scaler.fit_transform(df_fifa_numerico)

# Calcula a variância explicada acumulada
pca = PCA()
pca.fit(df_fifa_padronizado)
variancia_cumulativa = np.cumsum(pca.explained_variance_ratio_)

# Visualização da variância explicada acumulada
plt.plot(range(1, len(variancia_cumulativa) + 1), variancia_cumulativa, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Acumulada Explicada')
plt.title('Variância Acumulada Explicada pelo PCA')
# plt.show()


# Vamos definir um limiar de 80%, ou seja, queremos obter uma porcentagem de explicancia sobre
# nossos dados de igual a 80%
limiar_de_variancia = 0.80

# Encontrar o número de componentes necessários para atingir ou ultrapassar o limiar
num_de_pca = np.argmax(variancia_cumulativa >= limiar_de_variancia) + 1
print(f"Número de Componentes para {limiar_de_variancia * 100}% da Variância: {num_de_pca}")

# Inicializa o objeto PCA
pca = PCA(n_components=num_de_pca)
# Aplica o PCA aos dados padronizados
principal_components = pca.fit_transform(df_fifa_padronizado)

# Exibe a proporção de variância explicada
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)

# Pegando o número de componentes principais gerados
num_components = principal_components.shape[1]
# Gerando uma lista para cada PCA
columns_name = [f'PC{i}' for i in range(1, num_components + 1)]
# Criando um novo dataframe para visualizarmos como ficou nossos dados reduzidos com o PCA
pca_df = pd.DataFrame(data=principal_components, columns=columns_name)

# Criar histogramas para cada coluna
plt.figure(figsize=(15, 8))
for i, col in enumerate(pca_df.columns[:10]):
    plt.subplot(2, 5, i + 1)
    sns.histplot(pca_df[col], bins=20, kde=True)
    plt.title(f'Histograma {col}')

plt.tight_layout()
plt.show()

# Vamos olhar para cada coluna a normalidade após a redução de dimensionalidade
for column in pca_df.columns:
    stat, p_value = shapiro(pca_df[column])

    print(f'Variável: {column}, Estatística de teste: {stat}, Valor p: {p_value}')

    # Você pode então interpretar o valor p para determinar se a variável segue uma distribuição normal
    if p_value > 0.05:
        print(f'A variável {column} parece seguir uma distribuição normal.\n')
    else:
        print(f'A variável {column} não parece seguir uma distribuição normal.\n')