# Para processar arquivos e images
from PIL import Image
import glob
import numpy as np

# Para plotar imagens
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.cluster import KMeans

# Carregando as imagens
img_G = mpimg.imread('mdb001.pgm') # Tipo G
img_D = mpimg.imread('mdb003.pgm') # Tipo D
img_F = mpimg.imread('mdb005.pgm') # Tipo F

# Plotando as imagens
fig, axs = plt.subplots(1, 3, figsize=(10, 3))
im1 = axs[0].imshow(img_G, cmap='gray', vmin=0, vmax=255)
im1 = axs[1].imshow(img_D, cmap='gray', vmin=0, vmax=255)
im1 = axs[2].imshow(img_F, cmap='gray', vmin=0, vmax=255)
plt.show()

# Essa função usa o Kmeans como um filtro de segmentação de imagem
def filtro_kmeans(img, clusters):
    vectorized = img.reshape((-1, 1))
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=5)
    kmeans.fit(vectorized)

    centers = np.uint8(kmeans.cluster_centers_)
    segmented_data = centers[kmeans.labels_.flatten()]

    segmented__image = segmented_data.reshape((img.shape))
    return(segmented__image)

clusters = 3

img_G_segmentada = filtro_kmeans(img_G, clusters)
img_D_segmentada = filtro_kmeans(img_D, clusters)
img_F_segmentada = filtro_kmeans(img_F, clusters)

fig, axs = plt.subplots(1, 3, figsize=(10, 3))
im1 = axs[0].imshow(img_G_segmentada, cmap='gray', vmin=0, vmax=255)
im2 = axs[1].imshow(img_D_segmentada, cmap='gray', vmin=0, vmax=255)
im3 = axs[2].imshow(img_D_segmentada, cmap='gray', vmin=0, vmax=255)
plt.show()