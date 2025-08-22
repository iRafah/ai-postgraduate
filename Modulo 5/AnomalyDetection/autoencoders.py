import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Transformação dos dados para torch.FloatTensor e normalização dos valores (*)
# Se um pixel tem valor 0, ao dividir por 255, ainda será 0.
# Se um pixel tem valor 255, ao dividir por 255, vira 1.
# Um pixel que vale 128, por exemplo, ao dividir por 255, vira 0,5.
transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root = 'data',
    train = True,
    download = True,
    transform = transform
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    download = True,
    transform = transform
)

num_workers = 0
batch_size = 20

# Configuração dos data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# Obtendo uma amostra do batch de imagens de treino
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Convertendo para Numpy e Selecionando uma Imagem:
images = images.numpy()
img = np.squeeze(images[10])

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max() / 2.5

for x in range(width):
    for y in range(height):
        val = round(img[x][y], 2) if img[x][y] != 0 else 0
        ax.annotate(
            str(val), xy=(y, x),
            horizontalalignment='center',
            verticalalignment='center',
            color='white' if img[x][y] < thresh else 'black')

plt.show()