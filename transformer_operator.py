import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader

class SchrodingerDataset(Dataset):
    def __init__(self, data_dir):
        self.potentials_dir = os.path.join(data_dir, 'potentials')
        self.wavefuncs_dir = os.path.join(data_dir, 'wavefunctions')
        self.energies_dir = os.path.join(data_dir, 'energies')
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
        
        return V.unsqueeze(-1), wavefuncs, energies

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:x.size(1)].transpose(0,1).repeat(x.size(0), 1, 1)

class TransformerOperator(nn.Module):
    def __init__(self, n_points=1000, embed_dim=128, num_heads=8, num_layers=6):
        super().__init__()
        self.n_points = n_points
        
        # Encoder
        self.embedding = nn.Linear(1, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=n_points)
        
        # Transformer
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
        
    def forward(self, V):
        # V shape: (batch_size, n_points, 1)
        x = self.embedding(V)  # [batch, n_points, embed_dim]
        x = self.pos_encoder(x)  # Füge Positional Encoding hinzu
        x = self.transformer(x)  # [batch, n_points, embed_dim]
        
        # Energie-Vorhersage
        E_pred = self.energy_decoder(x.mean(dim=1))  # [batch, 5]
        
        # Wellenfunktions-Vorhersage
        psi_pred = self.wavefunc_decoder(x)  # [batch, n_points, 5]
        psi_pred = torch.nn.functional.normalize(psi_pred, dim=1)  # Normalisiere jede Wellenfunktion
        
        return E_pred, psi_pred
    
    def predict_solution(self, V, device='cpu'):
        self.eval()
        with torch.no_grad():
            V = torch.from_numpy(V).float().unsqueeze(0).unsqueeze(-1).to(device)
            E_pred, psi_pred = self(V)
            return E_pred.cpu().numpy(), psi_pred.cpu().numpy()


def train_model(model, train_loader, num_epochs=100, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Verlustfunktionen
    energy_criterion = nn.MSELoss()
    wavefunc_criterion = nn.MSELoss()
    
    # Physikalische Konstanten
    dx = 0.01  # Gitterabstand (in atomaren Einheiten)
    
    for epoch in range(num_epochs):
        
        print(epoch)
        
        model.train()
        total_loss = 0
        
        for V, wavefuncs, energies in train_loader:
            V = V.to(device)
            wavefuncs = wavefuncs.to(device)
            energies = energies.to(device)
            
            optimizer.zero_grad()
            E_pred, psi_pred = model(V)
            
            # 1. Energie-Verlust
            energy_loss = energy_criterion(E_pred, energies)
            
            # 2. Wellenfunktions-Verlust mit Normierungsbedingung
            wavefunc_loss = wavefunc_criterion(psi_pred, wavefuncs)
            
            # 3. Pauli-Prinzip -> Wie implementiere ich das? Die wellenfunktion muss ja schiefsymmetrisch sein

            # 4. Schrödinger-Gleichung als zusätzliche Bedingung
            schrodinger_loss = 0
            # Berechne zweite Ableitung mit finiten Differenzen
            d2_psi = (psi_pred[:, 2:, :] - 2*psi_pred[:, 1:-1, :] + psi_pred[:, :-2, :]) / (dx**2)
            # Potentialterm
            V_term = V[:, 1:-1, 0].unsqueeze(-1) * psi_pred[:, 1:-1, :]
            # Schrödinger-Gleichung: -ℏ²/2m * d²ψ/dx² + V*ψ = E*ψ
            # In atomaren Einheiten ist ℏ²/2m = 1
            schrodinger_residual = -d2_psi + V_term - E_pred.unsqueeze(1).expand(-1, psi_pred.shape[1]-2, -1) * psi_pred[:, 1:-1, :]
            schrodinger_loss = torch.mean(schrodinger_residual**2)
            
            # 5. Normierungsbedingung
            normalization_loss = 0
            for i in range(5):
                norm = torch.sum(psi_pred[:, :, i]**2, dim=1) * dx
                normalization_loss += torch.mean((norm - 1.0)**2)
            
            # 5. Energie-Ordnung (E_n+1 > E_n)
            # energy_order_loss = 0
            # for i in range(4):
            #     energy_order_loss += torch.mean(torch.relu(E_pred[:, i] - E_pred[:, i+1]))
            
            # Gesamtverlust mit Gewichtung
            loss = (0.3 * energy_loss + 
                   0.3 * wavefunc_loss + 
                   0.5 * schrodinger_loss +
                   0.1 * normalization_loss)
                   #0.05 * energy_order_loss)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}')
            print(f'  Energy Loss: {energy_loss.item():.4f}')
            print(f'  Wavefunc Loss: {wavefunc_loss.item():.4f}')
            print(f'  Schroedinger Loss: {schrodinger_loss.item():.4f}')
            print(f'  Normalization Loss: {normalization_loss.item():.4f}')


if __name__ == '__main__':
    # Beispiel für das Training
    dataset = SchrodingerDataset('training_data')
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = TransformerOperator()
    train_model(model, train_loader)
    
    # Speichere das trainierte Modell
    torch.save(model.state_dict(), 'transformer_operator.pth')