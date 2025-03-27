import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_operator import TransformerOperator
from solve_1d_systems import harmonic_potential, square_well_potential, double_well_potential

def extract_model_weights(model):
    """Extrahiert die Gewichte des Modells als NumPy-Arrays."""
    weights = {}
    
    # Extrahiere Embedding-Gewichte
    weights['embedding'] = model.embedding.weight.detach().numpy()
    weights['embedding_bias'] = model.embedding.bias.detach().numpy()
    
    # Extrahiere Transformer-Gewichte
    for i, layer in enumerate(model.transformer.layers):
        weights[f'transformer_layer_{i}'] = {
            'self_attn': {
                'weight': layer.self_attn.in_proj_weight.detach().numpy(),
                'bias': layer.self_attn.in_proj_bias.detach().numpy(),
                'out_weight': layer.self_attn.out_proj.weight.detach().numpy(),
                'out_bias': layer.self_attn.out_proj.bias.detach().numpy()
            },
            'linear1': {
                'weight': layer.linear1.weight.detach().numpy(),
                'bias': layer.linear1.bias.detach().numpy()
            },
            'linear2': {
                'weight': layer.linear2.weight.detach().numpy(),
                'bias': layer.linear2.bias.detach().numpy()
            }
        }
    
    # Extrahiere Decoder-Gewichte
    weights['energy_decoder'] = {
        'layer1_weight': model.energy_decoder[0].weight.detach().numpy(),
        'layer1_bias': model.energy_decoder[0].bias.detach().numpy(),
        'layer2_weight': model.energy_decoder[2].weight.detach().numpy(),
        'layer2_bias': model.energy_decoder[2].bias.detach().numpy()
    }
    
    weights['wavefunc_decoder'] = {
        'layer1_weight': model.wavefunc_decoder[0].weight.detach().numpy(),
        'layer1_bias': model.wavefunc_decoder[0].bias.detach().numpy(),
        'layer2_weight': model.wavefunc_decoder[2].weight.detach().numpy(),
        'layer2_bias': model.wavefunc_decoder[2].bias.detach().numpy()
    }
    
    return weights

def print_weight_shapes(weights):
    """Gibt die Formen aller Gewichte aus."""
    print("\nGewicht-Formen des Modells:")
    print("-" * 50)
    
    for name, weight in weights.items():
        if isinstance(weight, dict):
            print(f"\n{name}:")
            for subname, subweight in weight.items():
                print(f"  {subname}: {len(subweight)}")
        else:
            print(f"{name}: {len(weight)}")

def debug_model_outputs(model, V):
    """Debug-Funktion um die Ausgaben in verschiedenen Schichten zu überprüfen."""
    print("\nDebug-Ausgaben des Modells:")
    print("-" * 50)
    
    # Eingabe
    x = model.embedding(V)
    print(f"Nach Embedding - Form: {x.shape}")
    print(f"Nach Embedding - Wertebereich: [{x.min().item():.3f}, {x.max().item():.3f}]")
    
    # Nach jedem Transformer-Layer
    for i, layer in enumerate(model.transformer.layers):
        x = layer(x)
        print(f"\nNach Transformer Layer {i}:")
        print(f"Form: {x.shape}")
        print(f"Wertebereich: [{x.min().item():.3f}, {x.max().item():.3f}]")
        print(f"Standardabweichung: {x.std().item():.3f}")
        # Zeige die ersten paar Werte
        print(f"Erste 5 Werte: {x[0, :5, 0].detach().numpy()}")
    
    # Nach Mean-Pooling
    x_mean = x.mean(dim=1)
    print(f"\nNach Mean-Pooling:")
    print(f"Form: {x_mean.shape}")
    print(f"Wertebereich: [{x_mean.min().item():.3f}, {x_mean.max().item():.3f}]")
    print(f"Erste 5 Werte: {x_mean[0, :5].detach().numpy()}")
    
    # Energie-Decoder
    E_pred = model.energy_decoder(x_mean)
    print(f"\nEnergie-Decoder Ausgabe:")
    print(f"Form: {E_pred.shape}")
    print(f"Werte: {E_pred[0].detach().numpy()}")
    
    # Wellenfunktions-Decoder
    psi_pred = model.wavefunc_decoder(x)
    print(f"\nWellenfunktions-Decoder Ausgabe:")
    print(f"Form: {psi_pred.shape}")
    print(f"Wertebereich: [{psi_pred.min().item():.3f}, {psi_pred.max().item():.3f}]")
    print(f"Erste 5 Werte: {psi_pred[0, :5, 0].detach().numpy()}")
    
    # Überprüfe die Gewichte der Decoder
    print("\nDecoder-Gewichte:")
    print("Energie-Decoder Layer 1:")
    print(f"Gewichte: {model.energy_decoder[0].weight.data.numpy()}")
    print(f"Bias: {model.energy_decoder[0].bias.data.numpy()}")
    print("\nEnergie-Decoder Layer 2:")
    print(f"Gewichte: {model.energy_decoder[2].weight.data.numpy()}")
    print(f"Bias: {model.energy_decoder[2].bias.data.numpy()}")
    
    return E_pred, psi_pred

def visualize_prediction(x, V, E_pred, psi_pred, title="Vorhergesagte Lösung"):
    """Visualisiert die Vorhersagen des Modells."""
    plt.figure(figsize=(12, 8))
    
    # Konvertiere V von Tensor zu Array für das Plotten
    V_numpy = V.squeeze().detach().numpy()  # Entfernt die zusätzlichen Dimensionen
    
    # Plot Potential
    plt.plot(x, V_numpy, 'k-', label='Potential')
    
    # Plot Wellenfunktionen
    for n in range(5):
        # Konvertiere die Wellenfunktionen zu NumPy-Arrays
        psi_n = psi_pred[0, :, n].detach().numpy()  # [0] für erste Dimension, [:, n] für alle 1000 Einträge des n-ten Zustands
        E_n = E_pred[0, n].detach().numpy()  # Energie für den n-ten Zustand
        
        # Skaliere die Wellenfunktion für bessere Sichtbarkeit
        scale = 0.5  # Skalierungsfaktor
        plt.plot(x, E_n + scale * psi_n, 
                label=f'n={n}, E={E_n:.3f}')
    
    plt.title(title)
    plt.xlabel('x (a.u.)')
    plt.ylabel('Energie und Wellenfunktionen')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_and_visualize(model, V, x, title):
    """Macht Vorhersagen und visualisiert sie."""
    E_pred, psi_pred = debug_model_outputs(model, V)
    visualize_prediction(x, V, E_pred, psi_pred, title)

def main():
    # Lade das trainierte Modell
    model = TransformerOperator()
    model.load_state_dict(torch.load('transformer_operator.pth', map_location=torch.device('cpu')))
    model.eval()
    
    # Extrahiere und zeige die Gewichte
    weights = extract_model_weights(model)
    print_weight_shapes(weights)
    
    # Generiere x-Gitter
    x = np.linspace(-5, 5, 1000)
    
    # Beispiel 1: Harmonischer Oszillator
    V_harmonic = harmonic_potential(x)
    V_harmonic_tensor = torch.from_numpy(V_harmonic).float().unsqueeze(0).unsqueeze(-1)
    predict_and_visualize(model, V_harmonic_tensor, x, "Harmonischer Oszillator")
    
    # Beispiel 2: Rechteckiger Potentialtopf
    V_well = square_well_potential(x, depth=50, width=1)
    V_well_tensor = torch.from_numpy(V_well).float().unsqueeze(0).unsqueeze(-1)
    predict_and_visualize(model, V_well_tensor, x, "Rechteckiger Potentialtopf")
    
    # Beispiel 3: Doppelmuldenpotential
    V_double = double_well_potential(x, a=1, b=1)
    V_double_tensor = torch.from_numpy(V_double).float().unsqueeze(0).unsqueeze(-1)
    predict_and_visualize(model, V_double_tensor, x, "Doppelmuldenpotential")
    
    # Beispiel 4: Benutzerdefiniertes Potential
    V_custom = 0.5 * x**2 + 0.1 * np.sin(2 * x)  # Harmonischer Oszillator mit Störung
    V_custom_tensor = torch.from_numpy(V_custom).float().unsqueeze(0).unsqueeze(-1)
    predict_and_visualize(model, V_custom_tensor, x, "Benutzerdefiniertes Potential")

if __name__ == '__main__':
    main() 