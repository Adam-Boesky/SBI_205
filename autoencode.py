"""Autoencode and save the lightcurves"""
import sys
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.custom_sklearn import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('/Users/adamboesky/Research/SBI_205/models')

TRAIN = False


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 100)
        self.fc2 = nn.Linear(100, 300)
        self.fc3 = nn.Linear(300, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim=600, latent_dim=20):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder.forward(x)

    def decode(self, x):
        return self.decoder.forward(x)


def load_data():
    """Load the interpolated lightcurves."""
    # Import the LCs
    lcs = np.load('data/full_lcs_interped.npz', allow_pickle=True)['lcs']
    lcs = np.array([lc for lc in lcs if np.mean(lc.snrs) > 3])
    for lc in lcs:  # adjust the t explosion
        lc.theta[-1] -= min(lc.times)

    # Get the LCs
    X = np.array([np.concatenate((lc.mags_interped, lc.magerrs_interped)) for lc in lcs])

    return lcs, X


def train_auto_encoder(X_norm: np.ndarray):

    # Normalize the light curves and make a TensorDataset and DataLoader for batch processing
    X_norm_tensor = torch.tensor(X_norm, dtype=torch.float32)
    dataset = TensorDataset(X_norm_tensor)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Instantiate the model, loss function, and optimizer
    model = Autoencoder(input_dim=1200, latent_dim=25)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training loop
    num_epochs = 1000
    model.train()
    losses = []
    best_loss = float('inf')
    epochs_since_improvement = 0
    improvement_threshold = 100
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for data in data_loader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(data_loader.dataset)
        losses.append(epoch_loss)

        # Check for improvement
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Logging
        if (epoch+1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Auto-stop
        if epochs_since_improvement == improvement_threshold:
            print(f"No improvement in loss for {improvement_threshold} consecutive epochs, stopping training.")
            break
    model.eval()

    # Save the model
    torch.save(model, 'models/tuned_states/autoencoder0.pkl')

    return model


def encode_all_lightcurves():
    """Encode and save all of the lightcurves"""
    # Load data and normalize
    print('Loading data!!!')
    lcs, X = load_data()
    lc_scaler = StandardScaler()
    X_norm = lc_scaler.fit_transform(X)

    # Train
    print('Training!!!')
    if TRAIN:
        autoencoder = train_auto_encoder(X_norm)
    else:
        autoencoder = torch.load('models/tuned_states/autoencoder0.pkl')

    # Encode all the lightcurves and save data
    print('Encoding!!!')
    X_encoded = autoencoder(torch.tensor(X_norm, dtype=torch.float32)).detach().numpy()
    with open('data/full_encoded_lcs.pkl', 'wb') as f:
        pickle.dump((X_encoded, lcs), f)


if __name__=='__main__':
    encode_all_lightcurves()
