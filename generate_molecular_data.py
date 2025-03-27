import numpy as np
import os
from scipy.linalg import eigh
from solve_1d_systems import one_dim_solver



# Der Code da unten macht überhaupt keinen Sinn!!!
# Wir sollten die Moleküle nicht in einem 1D-System lösen, sondern in einem 3D-System.
# Außerdem können wir die Energieeigenwert aus Datenbanken bekommen
# Gibt es irgendeine Möglichkeit die Wellenfunktionen zu bekommen gegeben den Ene



def generate_diatomic_potential(x, R, Z1, Z2):
    """Generiert das effektive Potential für ein zweiatomiges Molekül."""
    # Coulomb-Potential zwischen den Kernen
    V_nn = Z1 * Z2 / R
    
    # Elektron-Kern-Potentiale
    r1 = np.abs(x + R/2)  # Abstand zum ersten Kern
    r2 = np.abs(x - R/2)  # Abstand zum zweiten Kern
    V_en = -Z1 / (r1 + 1e-10) - Z2 / (r2 + 1e-10)
    
    return V_en + V_nn

def generate_molecular_data(num_samples=1000, x_min=-10, x_max=10, N_points=1000):
    """Generiert Trainingsdaten für verschiedene zweiatomige Moleküle."""
    # Erstelle Verzeichnisse für die Daten
    os.makedirs('molecular_data', exist_ok=True)

    os.makedirs('molecular_data/potentials', exist_ok=True)
    os.makedirs('molecular_data/wavefunctions', exist_ok=True)
    os.makedirs('molecular_data/energies', exist_ok=True)
    
    # Liste von möglichen Atomkernen (Kernladungszahlen)
    nuclei = [(1,1), (1,2), (1,3), (2,2)]  # (H2, HD, HT, He2)
    
    # Generiere x-Gitter
    x = np.linspace(x_min, x_max, N_points)
    
    for i in range(num_samples):
        # Wähle zufällig ein Molekül
        Z1, Z2 = nuclei[np.random.randint(len(nuclei))]
        
        # Wähle zufälligen Kernabstand (in atomaren Einheiten)
        R = np.random.uniform(0.5, 3.0)
        
        # Generiere Potential
        V = generate_diatomic_potential(x, R, Z1, Z2)
        
        # Löse die Schrödinger-Gleichung
        energies, wavefuncs, _ = one_dim_solver(lambda x: V, x_min, x_max, N_points)
        
        # Speichere die Daten
        np.save(f'molecular_data/potentials/potential_{i}.npy', V)
        np.save(f'molecular_data/wavefunctions/wavefuncs_{i}.npy', wavefuncs[:, :5])
        np.save(f'molecular_data/energies/energies_{i}.npy', energies[:5])
        
        # Speichere zusätzliche Molekülparameter
        np.save(f'molecular_data/molecule_params_{i}.npy', np.array([Z1, Z2, R]))
        
        if (i + 1) % 100 == 0:
            print(f'Fortschritt: {i + 1}/{num_samples} Moleküle generiert')
    
    # Speichere x-Gitter
    np.save('molecular_data/x_grid.npy', x)
    
    print(f'Fertig! {num_samples} molekulare Trainingssamples wurden generiert und gespeichert.')

def visualize_molecule(V, wavefuncs, energies, x, Z1, Z2, R):
    """Visualisiert die Ergebnisse für ein einzelnes Molekül."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Plot Potential
    plt.plot(x, V, 'k-', label='Potential')
    
    # Plot Wellenfunktionen
    for n in range(5):
        plt.plot(x, energies[n] + wavefuncs[:, n], 
                label=f'n={n}, E={energies[n]:.3f}')
    
    plt.title(f'Molekül mit Z1={Z1}, Z2={Z2}, R={R:.2f}')
    plt.xlabel('x (a.u.)')
    plt.ylabel('Energie und Wellenfunktionen')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Generiere 1000 Trainingssamples
    generate_molecular_data(num_samples=1000)
    
    # Beispiel-Visualisierung
    x = np.load('molecular_data/x_grid.npy')
    V = np.load('molecular_data/potentials/potential_0.npy')
    wavefuncs = np.load('molecular_data/wavefuncs_0.npy')
    energies = np.load('molecular_data/energies_0.npy')
    params = np.load('molecular_data/molecule_params_0.npy')
    
    visualize_molecule(V, wavefuncs, energies, x, params[0], params[1], params[2]) 