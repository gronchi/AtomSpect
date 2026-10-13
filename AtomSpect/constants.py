"""Physical constants (SI unless stated)."""
import numpy as np



c_light = 299792458  # Speed of light in m/s
hbar = 1.0546e-34  # Plank's constant hbar in Angular
hh = 2.0*np.pi*hbar  # Plank's constant in frequency
muB = 9.27401e-24  # Bohr magneton J/T
muP = 2.79284734  # proton magneton (in terms of nuclear magnetons, nm)
muN = 1.5210322e-3  # Nucleon magneton (not nuclear dipole moment), in terms of muB
muN2 = muB/1836
mu_B = muB/hbar  # Common value of Bohr_Mag/hbar which ensures the computation doesn't drop small terms. [Hz/T]
g_L = 1  # Lande Orbital g-factor, assuming infinite mass, otherwise is g_L = 1 - me/mnuc
g_S = 2.002319  # Lande electron spin g-factor
a_o = 5.29e-11  # Bohr radii in m
mu_0 = 4*np.pi*1e-7  # freespace mag permeability
eps0 = 8.85e-12  # freespace permittivity
kboltz = 1.381e-23  # Boltzmann Constant
m_prot = 1.667e-27  # Mass of proton in kg
m_elec = 9.11e-31
q_charge = 1.602e-19
hhev = 4.135668e-15  # Plank's constant in Ev/Hz

muBEV = 5.7883818e-5  # bohr magneton in eV/Tesla
muBcm = muBEV/(1.23981e-4)  # bohr magneton in cm^-1
