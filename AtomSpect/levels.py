"""Construction of fine-structure and hyperfine state lists."""
import numpy as np

from .angular import CGw3, lande_gf, lande_gj, make_Jmj
from .hamiltonian import Zeemanlow
from .hfs import Delta_E_HFS

def _level_inputs(L, S, energies):
    """Validate angular momenta and one finite energy per ascending J."""
    for name, value in (('L', L), ('S', S)):
        if not np.isscalar(value) or not np.isfinite(value) or value < 0 or 2*value != round(2*value):
            raise ValueError(f'{name} must be a nonnegative integer or half integer')
    values = np.atleast_1d(np.asarray(energies, dtype=float))
    count = int(2*min(L, S) + 1)
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError(f'E_level requires {count} finite energies, lowest J first')
    return values

def _hfs_constants(values, count, name):
    """Broadcast a scalar HFS constant, or validate one value per J."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(count, values.item())
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError(f'{name} requires a scalar or {count} finite constants, lowest J first')
    return values

def _cg_matrix(states, match, coupled, target):
    """Matrix of Clebsch-Gordan coefficients <coupled| target> between basis states.

    Entry [i, j] couples state j's quantum numbers at positions `coupled`
    (j1, m1, j2, m2) into state i's `target` (J, M), and is zero unless the
    states agree at position `match` (the spectator quantum number).
    """
    dim = len(states)
    out = np.zeros((dim, dim))
    for i, state1 in enumerate(states):
        for j, state2 in enumerate(states):
            if state1[match] == state2[match]:
                out[i, j] = CGw3(*[state2[k] for k in coupled], state1[target[0]], state1[target[1]])
    return out


def _m_values(j, descending=False):
    """Magnetic quantum numbers -j..j as Python floats (ascending unless descending=True)."""
    m = [float(x) for x in np.arange(-j, j + 0.5)]
    return m[::-1] if descending else m


def _j_values(a, b):
    """Couplings |a-b|..a+b as Python floats."""
    return [float(x) for x in np.arange(abs(a - b), a + b + 0.1)]


def _sorted_by_low_field(states, energies, low_energies):
    """Basis order by weak-field energy; ties fall back to zero-field energy, then quantum numbers."""
    return sorted(range(len(states)), key=lambda i: (low_energies[i], energies[i], states[i]))


def stateprocess(L_state, S_state, E_level, thisB, sortE=False):
    """Build the coupled and uncoupled bases for one LS term.

    Parameters
    ----------
    L_state, S_state : float
        Nonnegative integer or half integer angular momenta; S is spin,
        not the multiplicity 2S+1.
    E_level : array-like
        One zero-field energy in cm^-1 per J, lowest J first.
    thisB : float
        Magnetic field in T, used only for the optional basis ordering.
    sortE : bool, optional
        Order the basis by its weak-field energies; default False.

    Returns
    -------
    list
        [Psi, E0_matrix, low_energies, E0_flat, states]. Psi transforms
        uncoupled columns to coupled rows; states are [J,mJ,L,mL,S,mS].
        low_energies is empty unless sortE=True. Energies are in cm^-1."""
    E_level = _level_inputs(L_state, S_state, E_level)
    L, S = float(L_state), float(S_state)
    J_values = _j_values(L, S)
    # Coupled |J,mJ> labels (lowest J and mJ first) are paired index by index with the
    # uncoupled |mL,mS> labels (highest mL, then mS, first); Psi relates the two bases.
    coupled = make_Jmj(J_values)
    uncoupled = [(mL, mS) for mL in _m_values(L, descending=True) for mS in _m_values(S, descending=True)]
    states = [[J, mJ, L, mL, S, mS] for (J, mJ), (mL, mS) in zip(coupled, uncoupled)]
    energies = np.repeat(E_level, [int(2*J + 1) for J in J_values]).tolist()

    low_energies = []
    if sortE:
        low = [Zeemanlow(lande_gj(st[0], st[2], st[4]), st[1], thisB) + E for st, E in zip(states, energies)]
        order = _sorted_by_low_field(states, energies, low)
        states = [states[i] for i in order]
        energies = [energies[i] for i in order]
        low_energies = [low[i] for i in order]

    # Couple (L,mL,S,mS) into (J,mJ) within each J manifold.
    Psi = _cg_matrix(states, match=2, coupled=(2, 3, 4, 5), target=(0, 1))
    return [Psi, np.diagflat(energies), low_energies, energies, states]


def stateprocess_HFS(L_state, S_state, I_state, HFS_Const, g_I, E_level, thisB, sortE=False):
    """Build the |F,mF> basis and coupling matrix for one hyperfine LS term.

    L_state, S_state and I_state are nonnegative angular momenta. E_level
    contains one energy in cm^-1 per J, lowest J first. HFS_Const contains
    HFSA and optional HFSB in Hz, each a scalar or one value per J.
    g_I follows lande_gi; thisB (T) is used only when sortE=True.

    Return [Psi, E0_matrix, HFS_shifts, fourth, states], with energies in
    cm^-1 and states [F,mF,J,mJ,I,mI,L,mL,S,mS]. Psi maps uncoupled columns
    to coupled rows. The legacy fourth entry is sorted weak-field energies
    when sortE=True, otherwise the expanded A constants in Hz."""
    E_level = _level_inputs(L_state, S_state, E_level)
    if not np.isscalar(I_state) or not np.isfinite(I_state) or I_state < 0 or 2*I_state != round(2*I_state):
        raise ValueError('I_state must be a nonnegative integer or half integer')
    L, S, I = float(L_state), float(S_state), float(I_state)
    J_values = _j_values(L, S)
    # Three label lists are paired index by index: |F,mF> (lowest F and mF first),
    # |J,mJ,I,mI> (mI ascending), and |mL,mS> repeated once per mI value (mL, mS ascending).
    coupled = make_Jmj([F for J in J_values for F in _j_values(J, I)])
    intermediate = [(J, mJ, mI) for J, mJ in make_Jmj(J_values) for mI in _m_values(I)]
    uncoupled = [(mL, mS) for mL in _m_values(L) for mS in _m_values(S) for _ in _m_values(I)]
    states = [[F, mF, J, mJ, I, mI, L, mL, S, mS]
              for (F, mF), (J, mJ, mI), (mL, mS) in zip(coupled, intermediate, uncoupled)]

    degeneracy = [int(2*J + 1)*int(2*I + 1) for J in J_values]
    energies = np.repeat(E_level, degeneracy).tolist()
    A_constants = np.repeat(_hfs_constants(HFS_Const['HFSA'], len(J_values), 'HFSA'), degeneracy).tolist()
    B_constants = np.repeat(_hfs_constants(HFS_Const.get('HFSB', 0), len(J_values), 'HFSB'), degeneracy).tolist()

    fourth = A_constants
    if sortE:
        low = [Zeemanlow(lande_gf(st[0], st[4], st[2], st[6], st[8], g_I), st[1], thisB) + E
               + Delta_E_HFS(I, st[2], st[0], A, B)
               for st, E, A, B in zip(states, energies, A_constants, B_constants)]
        order = _sorted_by_low_field(states, energies, low)
        states = [states[i] for i in order]
        energies = [energies[i] for i in order]
        A_constants = [A_constants[i] for i in order]
        B_constants = [B_constants[i] for i in order]
        # The legacy fourth entry is the sorted weak-field energies here, the A constants otherwise.
        fourth = [low[i] for i in order]

    hfs_shifts = [Delta_E_HFS(st[4], st[2], st[0], A, B) for st, A, B in zip(states, A_constants, B_constants)]
    # Couple (J,mJ) with (I,mI) into (F,mF) within each J manifold, and (L,mL) with (S,mS)
    # into (J,mJ) at fixed mI.
    Psi = (_cg_matrix(states, match=2, coupled=(2, 3, 4, 5), target=(0, 1))
           @ _cg_matrix(states, match=5, coupled=(6, 7, 8, 9), target=(2, 3)))
    return [Psi, np.diagflat(energies), hfs_shifts, fourth, states]
