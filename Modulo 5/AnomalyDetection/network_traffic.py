import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Aplicando o PCA e visualizando os dados
np.random.seed(42)

normal_traffic = np.random.normal(loc=50, scale=10, size=(100, 3))
anomalous_traffic = np.array([[120, 200, 80], [5, 10, 5]])
network_traffic = np.vstack([normal_traffic, anomalous_traffic])

# Aplicando o PCA e visualizando os dados
pca = PCA(n_components=2)
pca_result = pca.fit_transform(network_traffic)
explained_variance = pca.explained_variance_ratio_

print(f"Variância explicada por cada componente principal: {explained_variance}")

plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:-2, 0], pca_result[:-2, 1], label='Tráfego Normal', alpha=0.7, color='b')
plt.scatter(pca_result[-2:, 0], pca_result[-2:, 1], color='r', label='Anomalias', marker='x', s=100)
plt.title('Redução Dimensional com PCA (Tráfego de Rede)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid(alpha=0.5)
plt.show()