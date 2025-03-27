import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Solver for 1D Schrödinger equation
def one_dim_solver(V, x_min, x_max, N):
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]

    # Kinetic energy operator discretization
    diagonal = (1 / dx**2) + V(x)
    off_diagonal = -0.5 / dx**2 * np.ones(N - 1)

    # Solve tridiagonal eigenvalue problem
    energies, wavefuncs = eigh_tridiagonal(diagonal, off_diagonal)

    # Normalize wavefunctions
    wavefuncs = wavefuncs / np.sqrt(dx)

    return energies, wavefuncs, x

# Example potentials:
def harmonic_potential(x):
    return 0.5 * x**2

def square_well_potential(x, depth=50, width=1):
    V = np.zeros_like(x)
    V[np.abs(x) > width] = depth
    return V

def infinite_square_well(x, width=1):
    V = np.zeros_like(x)
    V[np.abs(x) > width/2] = 1e6  # Sehr hoher Wert simuliert unendlich
    return V

def double_well_potential(x, a=1, b=1):
    return a * (x**2 - b)**2

def gaussian_potential(x, V0=-50, sigma=0.5):
    return V0 * np.exp(-x**2 / (2 * sigma**2))

def linear_potential(x, slope=10):
    return slope * x

def exponential_potential(x, V0=20, alpha=1):
    return V0 * np.exp(alpha * np.abs(x))

def coulomb_like_potential(x, Z=1, eps=1e-2):
    return -Z / (np.abs(x) + eps)  # eps verhindert Division durch 0

def sinusoidal_potential(x, V0=10, k=2):
    return V0 * np.sin(k * x)

# Function to generate random potentials
def random_potential_generator(num_potentials, x):
    potentials = []
    for _ in range(num_potentials):
        pot_type = np.random.choice(['harmonic', 'square_well', 'double_well', 'gaussian'])
        if pot_type == 'harmonic':
            omega = np.random.uniform(0.5, 2.0)
            potentials.append(0.5 * omega**2 * x**2)
        elif pot_type == 'square_well':
            depth = np.random.uniform(20, 100)
            width = np.random.uniform(0.5, 2.0)
            V = np.zeros_like(x)
            V[np.abs(x) > width] = depth
            potentials.append(V)
        elif pot_type == 'double_well':
            a = np.random.uniform(0.5, 2.0)
            b = np.random.uniform(0.5, 2.0)
            potentials.append(a * (x**2 - b)**2)
        elif pot_type == 'gaussian':
            V0 = np.random.uniform(-100, -10)
            sigma = np.random.uniform(0.3, 1.0)
            potentials.append(V0 * np.exp(-x**2 / (2 * sigma**2)))
    return potentials


# # Parameters
# x_min, x_max = -5, 5
# N_points = 1000

# # Solve Schrödinger equation for harmonic oscillator
# energies, wavefuncs, x = one_dim_solver(square_well_potential, x_min, x_max, N_points)

# # Plot first three eigenstates
# plt.figure(figsize=(10,6))
# plt.plot(x, harmonic_potential(x), 'k-', label='Potential')
# for n in range(4):
#     plt.plot(x, energies[n] + wavefuncs[:, n], label=f'n={n}, E={energies[n]:.3f}')

# plt.xlabel('x')
# plt.ylabel('Energy and Wavefunctions')
# plt.title('Eigenstates of 1D Schrödinger Equation (Harmonic Oscillator)')
# plt.legend()
# plt.grid()
# plt.show()