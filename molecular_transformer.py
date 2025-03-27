import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class MolecularDataset(Dataset):
    def __init__(self, data_dir):
        self.potentials_dir = os.path.join(data_dir, 'potentials')
        self.wavefuncs_dir = os.path.join(data_dir, 'wavefunctions')
        self.energies_dir = os.path.join(data_dir, 'energies')
        self.params_dir = os.path.join(data_dir, 'molecule_params')
        self.num_samples = len(os.listdir(self.potentials_dir))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Lade Potential
        V = torch.from_numpy(np.load(os.path.join(self.potentials_dir, f'potential_{idx}.npy'))).float()
        
        # Lade Wellenfunktionen (erste 5 Eigenzustände)
        wavefuncs = torch.from_numpy(np.load(os.path.join(self.wavefuncs_dir, f'wavefuncs_{idx}.npy'))).float()
        
        # Lade Energien (erste 5 Eigenwerte)
        energies = torch.from_numpy(np.load(os.path.join(self.energies_dir, f'energies_{idx}.npy'))).float()
        
        # Lade Molekülparameter (Z1, Z2, R)
        params = torch.from_numpy(np.load(os.path.join(self.params_dir, f'molecule_params_{idx}.npy'))).float()
        
        return V.unsqueeze(-1), wavefuncs, energies, params

class MolecularTransformer(nn.Module):
    def __init__(self, n_points=1000, embed_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.n_points = n_points
        
        # Encoder für das Potential
        self.embedding = nn.Linear(1, embed_dim)
        
        # Encoder für Molekülparameter
        self.param_embedding = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Transformer für das Potential
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Decoder für Energien
        self.energy_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 5)  # 5 Energieniveaus
        )
        
        # Decoder für Wellenfunktionen
        self.wavefunc_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 5)  # 5 Wellenfunktionen
        )
        
    def forward(self, V, params):
        # V shape: (batch_size, n_points, 1)
        # params shape: (batch_size, 3) - [Z1, Z2, R]
        
        # Verarbeite Potential
        x = self.embedding(V)  # [batch, n_points, embed_dim]
        
        # Verarbeite Molekülparameter
        param_features = self.param_embedding(params)  # [batch, embed_dim]
        
        # Füge Parameter-Information zum Potential hinzu
        x = x + param_features.unsqueeze(1)
        
        # Transformer-Verarbeitung
        x = self.transformer(x)  # [batch, n_points, embed_dim]
        
        # Energie-Vorhersage
        E_pred = self.energy_decoder(x.mean(dim=1))  # [batch, 5]
        
        # Wellenfunktions-Vorhersage
        psi_pred = self.wavefunc_decoder(x)  # [batch, n_points, 5]
        psi_pred = torch.nn.functional.normalize(psi_pred, dim=1)  # Normalisiere jede Wellenfunktion
        
        return E_pred, psi_pred

def train_model(model, train_loader, num_epochs=100, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    energy_criterion = nn.MSELoss()
    wavefunc_criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for V, wavefuncs, energies, params in train_loader:
            V = V.to(device)
            wavefuncs = wavefuncs.to(device)
            energies = energies.to(device)
            params = params.to(device)
            
            optimizer.zero_grad()
            E_pred, psi_pred = model(V, params)
            
            # Berechne Verluste
            energy_loss = energy_criterion(E_pred, energies)
            wavefunc_loss = wavefunc_criterion(psi_pred, wavefuncs)
            loss = energy_loss + wavefunc_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

def predict_solution(model, V, params, device='cuda'):
    model.eval()
    with torch.no_grad():
        V = torch.from_numpy(V).float().unsqueeze(0).unsqueeze(-1).to(device)
        params = torch.from_numpy(params).float().unsqueeze(0).to(device)
        E_pred, psi_pred = model(V, params)
        return E_pred.cpu().numpy(), psi_pred.cpu().numpy()

if __name__ == '__main__':
    # Beispiel für das Training
    dataset = MolecularDataset('molecular_data')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MolecularTransformer()
    train_model(model, train_loader)
    
    # Speichere das trainierte Modell
    torch.save(model.state_dict(), 'molecular_transformer.pth') 