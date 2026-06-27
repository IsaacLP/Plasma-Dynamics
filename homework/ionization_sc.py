# Author: Isaac López

import numpy as np

# Constants
m_e = 9.10938356e-31  # Electron mass (kg)
h = 6.62607015e-34  # Planck's constant (J·s)
k_B = 1.380649e-23  # Boltzmann constant (J/K)

T_sc = 1.5e7  # Temperature of the solar core (K)
n_sc = 1e32 # Number density of particles in the solar core (particles/m^3)
E_j = 13.6  # Ionization energy of hydrogen (eV)

E_j = E_j * 1.60218e-19  # Convert eV to Joules

frac = (2*np.pi*m_e)**(3/2) / h**3 * (k_B * T_sc)**(5/2)/(n_sc * k_B * T_sc) * np.exp(-E_j / (k_B * T_sc))
alpha = np.sqrt(frac / (1 + frac))
print(f"Fraction of ionized hydrogen in the solar core: {alpha:.4f}")