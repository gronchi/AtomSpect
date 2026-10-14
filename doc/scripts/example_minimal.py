"""Minimal example: Zeeman pattern of a He I line with polarization and a convolved spectrum.

Run from the repository root:  python doc/scripts/example_minimal.py
"""
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault('MPMATH_NOGMPY', '1')
os.environ.setdefault('SYMPY_GROUND_TYPES', 'python')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import AtomSpect

deck = dict(
    s_ground=1, s_excited=1, l_ground=1, l_excited=2,      # 1s2p 3P -> 1s4d 3D (S is the spin quantum number)
    E_ground=np.array([169087.8308131, 169086.8428979, 169086.7664725]),     # cm^-1, lowest J first
    E_excited=np.array([191444.49804915, 191444.47952984, 191444.47832914]),
    Bmag=0.5, b_angle=90,                                   # field in T and angle to the line of sight
    amu=4, Temp=300, Convfxn='Skewed', Skewness=[1.24, 1.98],
    spec_window=[446.9, 447.4], plot_window=[447.0, 447.3], specstep=0.002222, specres=3,
)

result = AtomSpect.AtomSpect_Main(deck)
assert np.isclose(np.sum(result.signal), 1)
assert np.all(np.isfinite(result.SpecOut))
assert np.all(np.diff(result.SpecOut[0]) >= 0)

strongest = np.argsort(result.signal)[::-1][:5]
print(f"{len(result.signal)} eigenstate pairs; strongest five (wavelength in air, relative strength, q):")
for i in strongest:
    print(f"  {result.wave_air[i]:.4f} nm  {result.signal[i]:.4f}  q={result.rad[i][1]}")

wavelength, intensity = result.SpecOut
print(f"convolved spectrum: {len(wavelength)} points, peak at {wavelength[np.argmax(intensity)]:.3f} nm")

if '--plot' in sys.argv:
    from matplotlib import pyplot as plt
    plt.plot(wavelength, intensity/np.max(intensity))
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized intensity')
    plt.show()
