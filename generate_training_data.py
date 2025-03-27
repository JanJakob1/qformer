import numpy as np
import os
from solve_1d_systems import one_dim_solver, random_potential_generator

def generate_and_save_training_data(num_samples=1000, x_min=-5, x_max=5, N_points=1000):
    # Erstelle Verzeichnisse für die Daten
    os.makedirs('training_data', exist_ok=True)
    os.makedirs('training_data/potentials', exist_ok=True)
    os.makedirs('training_data/wavefunctions', exist_ok=True)
    os.makedirs('training_data/energies', exist_ok=True)
    
    # Generiere x-Gitter
    x = np.linspace(x_min, x_max, N_points)
    
    # Generiere zufällige Potentiale
    potentials = random_potential_generator(num_samples, x)
    
    # Speichere die Daten
    for i in range(num_samples):
        # Löse die Schrödinger-Gleichung für jedes Potential
        energies, wavefuncs, _ = one_dim_solver(lambda x: potentials[i], x_min, x_max, N_points)
        
        # Speichere Potential
        np.save(f'training_data/potentials/potential_{i}.npy', potentials[i])
        
        # Speichere Wellenfunktionen (erste 5 Eigenzustände)
        np.save(f'training_data/wavefunctions/wavefuncs_{i}.npy', wavefuncs[:, :5])
        
        # Speichere Energien (erste 5 Eigenwerte)
        np.save(f'training_data/energies/energies_{i}.npy', energies[:5])
        
        if (i + 1) % 100 == 0:
            print(f'Fortschritt: {i + 1}/{num_samples} Samples generiert')
    
    # Speichere x-Gitter für spätere Verwendung
    np.save('training_data/x_grid.npy', x)
    
    print(f'Fertig! {num_samples} Trainingssamples wurden generiert und gespeichert.')

if __name__ == '__main__':
    # Generiere 1000 Trainingssamples
    generate_and_save_training_data(num_samples=1000) 