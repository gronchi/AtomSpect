"""Axial dc Stark Hamiltonian: Steck Rb-87 Eqs. (30)-(32).

Electric fields are V/cm; polarizabilities are alpha/h in Hz/(V/cm)^2.
Only scalar and diagonal-electronic-J tensor polarizabilities are represented.
"""
import numpy as np

from .angular import CGCoeff
from .constants import c_light

__all__ = ['stark_hamiltonian']


def _electric_field(value):
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError('Efield must be a finite scalar in V/cm')
    return float(value)


def _polarizabilities(values, j_values, name, default=None):
    if values is None:
        if default is None:
            raise ValueError(f'{name} is required for nonzero Efield')
        values = default
    array = np.asarray(values, dtype=float)
    if array.ndim > 1 or array.size not in (1, len(j_values)) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} requires a finite scalar or one value per J, lowest J first')
    return dict(zip(j_values, np.broadcast_to(array.reshape(-1), (len(j_values),))))


def _stark_shifts(j, m, electric_field, alpha0, alpha2, c=c_light):
    """Diagonal |J,mJ> energy shifts in cm^-1, with alpha/h inputs."""
    j, m = np.asarray(j, float), np.asarray(m, float)
    a0, a2 = np.asarray(alpha0, float), np.asarray(alpha2, float)
    allowed = j >= 1
    if np.any((~allowed) & (a2 != 0)):
        raise ValueError('Tensor polarizability must be zero for J < 1')
    factor = np.zeros_like(j)
    factor[allowed] = (3*m[allowed]**2-j[allowed]*(j[allowed]+1))/(j[allowed]*(2*j[allowed]-1))
    return -.5*electric_field**2*(a0+a2*factor)/(100*c)


def stark_hamiltonian(level, Efield=0., alpha0=None, alpha2=None):
    """Return the dc Stark matrix in a stateprocess[_HFS] coupled basis.

    Efield is a signed axial electric field in V/cm. alpha0 and alpha2 are
    alpha/h in Hz/(V/cm)^2, scalar or one per electronic J (lowest J first).
    A nonzero field requires alpha0; omitted alpha2 means zero. Fields must
    lie along the quantization axis. No cross-J Stark tensors are inferred.
    A zero field needs no polarizability data and returns a zero matrix.
    """
    electric = _electric_field(Efield)
    states = level[-1]
    if electric == 0:
        return np.zeros((len(states), len(states)))
    hfs = len(states[0]) == 10
    js = np.array([s[2] if hfs else s[0] for s in states])
    ms = np.array([s[3] if hfs else s[1] for s in states])
    j_values = sorted(set(js))
    scalar = _polarizabilities(alpha0, j_values, 'alpha0')
    tensor = _polarizabilities(alpha2, j_values, 'alpha2', default=0.)
    shifts = _stark_shifts(js, ms, electric, [scalar[j] for j in js], [tensor[j] for j in js])
    if not hfs:
        return np.diag(shifts)
    # Couple only J and I, rather than using level[0]'s full L/S/I transform.
    U = np.array([[CGCoeff(b[2], b[3], b[4], b[5], a[0], a[1])
                   if a[2] == b[2] else 0. for b in states] for a in states])
    return (U*shifts) @ U.T
