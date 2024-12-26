from sklearn.datasets import fetch_openml # import the open ML
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV #metodo para seleção dos melhores Ks (#Basicamente a busca por força bruta)
from sklearn.metrics import make_scorer, accuracy_score, f1_score #métricas de validação
from sklearn.ensemble import RandomForestClassifier # RandomForest
from sklearn.svm import SVC  
from sklearn.metrics import ConfusionMatrixDisplay

dados = fetch_openml(data_id=1523)
tabela_dados = pd.DataFrame(data=dados['data'])
print(tabela_dados.head())

# Tranform the database
classes = {'1':'Disk Hernia',
           '2':'Normal',
           '3':'Spondylolisthesis'}

tabela_dados['diagnostic'] = [classes[target] for target in dados.target]
print(tabela_dados.tail())

# Exploratory data analysis
print(tabela_dados.info())

print(len(tabela_dados))

print(tabela_dados.describe())

# What's the mean of the data?
print(tabela_dados.groupby('diagnostic').mean())

print(tabela_dados.groupby('diagnostic').describe())

# Any outlier?
print(tabela_dados.loc[tabela_dados['V6'] > 400])

# Remove outlier
tabela_dados.drop(tabela_dados.loc[tabela_dados['V6'] > 400].index, inplace=True)
print(tabela_dados.loc[tabela_dados['V6'] > 400])

x = tabela_dados.drop(columns=['diagnostic'])
y = tabela_dados['diagnostic'] # target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

# Normalizing the data
scaler = MinMaxScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

modelo_classificador = KNeighborsClassifier(n_neighbors=3)
modelo_classificador.fit(x_train_scaled, y_train)

y_predicto = modelo_classificador.predict(x_test_scaled)
tabela_dados.groupby('diagnostic').count()

matriz_confusao = confusion_matrix(y_true = y_test,
                                   y_pred = y_predicto,
                                   labels=['Disk Hernia', 'Normal', 'Spondylolisthesis'])

figure = plt.figure(figsize=(15, 5))
disp = ConfusionMatrixDisplay(confusion_matrix = matriz_confusao,
                              display_labels=['Disk Hernia', 'Normal', 'Spondylolisthesis'])

disp.plot(values_format='d')

# plt.show()
kfold  = KFold(n_splits=5, shuffle=True)  # shuffle=True, Shuffle (embaralhar) os dados.
result = cross_val_score(modelo_classificador, x, y, cv = kfold)

print("K-Fold (R^2) Scores: {0}".format(result))
print("Mean R^2 for Cross-Validation K-Fold: {0}".format(result.mean()))

error = []

for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_scaled, y_train)
    pred_i = knn.predict(x_test_scaled)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 15), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
# plt.show()

# Search for the best parameters
# We will use a technique called Gridsearch, which is basically a brute force search
# We will use the 5-fold cross-validation technique (divisions) on the training set
# As the best performance metric, we will use accuracy, that is, we are looking for the hyperparameters that maximize accuracy

# Tested parameters
param_grid = {
    'n_neighbors': [8, 14], # total of neighbors
    'weights': ['uniform', 'distance'], 
    'metric': ['cosine', 'euclidean', 'manhattan']
}

gs_metric = make_scorer(accuracy_score, greater_is_better=True)
grid = GridSearchCV(KNeighborsClassifier(),
                    param_grid=param_grid,
                    scoring=gs_metric,
                    cv=5, n_jobs=4, verbose=3)

grid.fit(x_train_scaled, y_train)
knn_params = grid.best_params_
print('KNN', knn_params)

print(grid.cv_results_)

def AplicaValidacaoCruzada(x_axis, y_axis):
    # Linear models.
    from sklearn.neighbors import KNeighborsClassifier  # k-vizinhos mais próximos (KNN)
    from sklearn.ensemble import RandomForestClassifier # RandomForest
    from sklearn.svm import SVC                         # Maquina de Vetor Suporte SVM

    # Cross-Validation models.
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    kfold = KFold(n_splits=10, shuffle=True)
    # Axis
    x = x_axis
    y = y_axis

    # KNN
    knn = KNeighborsClassifier(n_neighbors=8, metric='euclidean', weights='distance')
    knn.fit(x_train_scaled, y_train)

    # SVM
    svm = SVC()
    svm.fit(x_train_scaled, y_train)

    # RandomForest
    rf = RandomForestClassifier(random_state=7)
    rf.fit(x_train_scaled, y_train)

    knn_result = cross_val_score(knn, x, y, cv = kfold)
    svm_result = cross_val_score(svm, x, y, cv = kfold)
    rf_result = cross_val_score(rf, x, y, cv = kfold)

    dic_models = {
        'KNN': knn_result.mean(),
        'SVM': svm_result.mean(),
        'RF': rf_result.mean()
    }

    # Select the best model.
    melhorModelo = max(dic_models, key=dic_models.get)
    
    print("KNN (R^2): {0}\nSVM (R^2): {1}\nRandom Forest (R^2): {2}".format(knn_result.mean(), svm_result.mean(), rf_result.mean()))
    print("O melhor modelo é : {0} com o valor: {1}".format(melhorModelo, dic_models[melhorModelo]))

AplicaValidacaoCruzada(x, y)