import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Simulate data
np.random.seed(0)
normal_data = np.random.normal(7, 0.5, (1000, 10))
anomaly_data_low = np.random.normal(3, 0.5, (25, 10))
anomaly_data_high = np.random.normal(11, 0.5, (25, 10))

# Combine and convert data
data = np.concatenate([normal_data, anomaly_data_low, anomaly_data_high], axis=0)
data = torch.tensor(data, dtype=torch.float32)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 6),
            nn.ReLU(),
            nn.Linear(6, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 10)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 100
for epochs in range(epochs):
    output = model(data)
    loss = criterion(output, data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    reconstructed_data = model(data)
    mse = ((reconstructed_data - data) ** 2).mean(axis=1)

threshold = mse[:1000].mean() + 3 * mse[:1000].std()
anomalies = mse > threshold

plt.figure(figsize=(10, 5))
plt.hist(mse[:1000], bins=50, alpha=0.6, label="Normal Data")
plt.hist(mse[1000:], bins=50, alpha=0.6, label="Anomaly Data")
plt.axvline(threshold, color='red', linestyle='--', label="Threshold")
plt.legend()
plt.xlabel("Reconstruction Error")
plt.ylabel("Frequency")
plt.title("Anomaly Detection using Autoencoder - pH Analysis")
plt.show()