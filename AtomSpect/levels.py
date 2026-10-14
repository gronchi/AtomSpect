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
    # Create an array that contains [L,ml,S,ms] for the given L,S values. Starts with highest ml,mS values and itereates through ms first.
    LS0 = [[float(L_state), float(x), float(S_state), float(y)] for x in np.linspace(L_state, -L_state, int(2*L_state+1), dtype=float) for y in np.linspace(S_state, -S_state, int(2*S_state+1), dtype=float)]
    # LS0 = [[L_state, x, S_state, y] for x in np.linspace(L_state, -L_state, int(2*L_state+1),dtype=float) for y in np.linspace(S_state, -S_state, int(2*S_state+1),dtype=float) ]
    # print(LS0)
    # Now create the possible J values
    Jvals = np.arange(np.abs(L_state-S_state), (L_state+S_state+.1))  # ,dtype=float)

    # Use those J values to make all possible mJ values and starts with Lowest J, and lowest mJ
    Jmjs0 = make_Jmj(Jvals)

    # Combine both lists together to be a single large [J,mJ,L,mL,S,mS] list. The dimensions should match for LS0 and Jmjs0.
    bigstatelist = [[x[0], x[1], y[0], y[1], y[2], y[3]] for x, y in zip(Jmjs0, LS0)]

    # Expand the input energies which were given for each J to correspond to the number of mJs
    E_cm0 = [[E_level[x]]*int(2*Jvals[x] + 1) for x in range(len(E_level))]
    # Flatten the results out
    E_cm0_flat = [item for row in E_cm0 for item in row]
    # This step uses a low field approximation in order to sort the state-list based on the given magnetic field such that the relative state function order is known.
    # This is used for signal strength calculation as otherwise we may not know what order the functions are after they get diagonalized. This step might not be needed.
    LowE = []

    if sortE == True:
        biglist = []
        for i, x in enumerate(E_cm0_flat):
            thislandegJ = lande_gj(bigstatelist[i][0], bigstatelist[i][2], bigstatelist[i][4])  # caluclate lande gj factor for a given state
            if thisB == 0:
                thisZeemanEnergy = x  # This is to prevent issues with attempting to sort for B=0
            else:
                thisZeemanEnergy = Zeemanlow(thislandegJ, bigstatelist[i][1], thisB) + x  # This is the low zeeman approximation total energy
            biglist.append([thisZeemanEnergy, x, Jmjs0[i], LS0[i], bigstatelist[i]])

        sortedlist = sorted(biglist)  # Soted in ascending energy
        LowE = [x[0] for x in sortedlist]
        states = [x[-1] for x in sortedlist]
        E_flat = [x[1] for x in sortedlist]  # This is the sorted energies, and is in cm^-1
    else:
        states = bigstatelist
        E_flat = E_cm0_flat

    # States are [J,mJ,L,mL,S,mS]: couple (L,mL,S,mS) into (J,mJ) within each J manifold.
    w3mat = _cg_matrix(states, match=2, coupled=(2, 3, 4, 5), target=(0, 1))
    E_mat = np.diagflat(E_flat)  # Despite the old name, these energies are in cm^-1
    return [w3mat, E_mat, LowE, E_flat, states]


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
    # Create an array that contains [L,ml,S,ms] for the given L,S values. Starts with lowest ml,mS values and itereates through ms first.
    # LS0 = [[L_state, x, S_state, y] for x in np.linspace(-L_state, L_state, int(2*L_state+1)) for y in np.linspace(-S_state, S_state, int(2*S_state+1)) ]
    # Create an array that contains [L,ml,S,ms,I,mI] for the given L,S,I values. Starts with Highest mL,mS values and itereates through mI first, then mS, then mL
    # Extra type-casting is required due to new numpy.float64 objects interaction with sympy wigner functions
    LSI0 = [[float(L_state), float(x), float(S_state), float(y), float(I_state), float(z)] for x in np.linspace(-L_state, L_state, int(2*L_state+1)) for y in np.linspace(-S_state, S_state, int(2*S_state+1))for z in np.linspace(I_state, -I_state, int(2*I_state+1))]

    # Now create the V values, starting with the lowest J state first.

    # Make all Js corresponding to L,S relation
    Jvals = np.arange(np.abs(L_state-S_state), (L_state+S_state+.1))

    # Make all mJs for the Js
    Jmjs0 = make_Jmj(Jvals)

    # Make a list of J,mJ,I,mI states
    JI0 = [[float(x[0]), float(x[1]), float(I_state), float(y)] for x in Jmjs0 for y in np.linspace(-I_state, I_state, int(2*I_state+1))]

    # Combine J and Is to make possible Fs
    Fvals = [np.arange(np.abs(-x + I_state), (x + I_state+.1)) for x in Jvals]
    # Flatten the list to be one big list
    Fvals = [float(item) for row in Fvals for item in row]
    # Make the possible mFs for all Fs
    Fmfs0 = make_Jmj(Fvals)

    E_cm0 = [[E_level[x]]*int(2*Jvals[x] + 1)*int(2*I_state + 1) for x in range(len(E_level))]
    E0_cm_flat = [item for row in E_cm0 for item in row]

    HFS_A_In = _hfs_constants(HFS_Const['HFSA'], len(Jvals), 'HFSA')
    HFS_B_In = _hfs_constants(HFS_Const.get('HFSB', 0), len(Jvals), 'HFSB')
    degeneracy = [int(2*j + 1)*int(2*I_state + 1) for j in Jvals]
    HFS_Flat = np.repeat(HFS_A_In, degeneracy).tolist()
    HFS_FlatB = np.repeat(HFS_B_In, degeneracy).tolist()
    # Duplicate energies based on degeneracy and flatten the results

    # Duplicate HFS constants to match F levels based on degeneracy and flatten the results
    # HFSConst = [[HFS_Const[0][x]]*int(2*Jvals[x] +1)*int(2*I_state + 1) for x in range(len(E_level))]
    # Flatten the shape of the Energy array

    # Make a large list of [F,mF,J,mJ,I,mI,L,mL,S,mS]
    bigstatelist = [[x[0], x[1], y[0], y[1], y[2], y[3], z[0], z[1], z[2], z[3]] for x, y, z, in zip(Fmfs0, JI0, LSI0)]

    # This step uses a low field approximation in order to sort the state-list based on the given magnetic field such that the relative state function order is known.
    # This is used for signal strength calculation as otherwise we may not know what order the functions are after they get diagonalized. This step might not be needed.

    if sortE:
        biglist = []
        for i, x in enumerate(E0_cm_flat):
            thislandegF = lande_gf(bigstatelist[i][0], bigstatelist[i][4], bigstatelist[i][2], bigstatelist[i][6], bigstatelist[i][8], g_I)  # Calculate lande gf factor for the given transition

            thisZeemanEnergy = Zeemanlow(thislandegF, bigstatelist[i][1], thisB) + x + Delta_E_HFS(I_state, bigstatelist[i][2], bigstatelist[i][0], HFS_Flat[i], HFS_FlatB[i])

            biglist.append([thisZeemanEnergy, x, bigstatelist[i], HFS_Flat[i], HFS_FlatB[i]])
        sortedlist = sorted(biglist)
        states = [x[2] for x in sortedlist]
        sortedHFC = [x[3] for x in sortedlist]  # Sorted hyperfine constants
        sortedHFCB = [x[4] for x in sortedlist]  # Sorted hyperfine constants
        E_cm = [x[1] for x in sortedlist]
        # The fourth entry is the sorted low-field energies here, but the unsorted
        # constants below (kept as the original returned them).
        fourth = [x[0] for x in sortedlist]
    else:
        states = bigstatelist
        sortedHFC, sortedHFCB = HFS_Flat, HFS_FlatB
        E_cm = E0_cm_flat
        fourth = HFS_Flat

    E_HFSFlat = [Delta_E_HFS(st[4], st[2], st[0], sortedHFC[i], sortedHFCB[i]) for i, st in enumerate(states)]  # Hyperfine + E0 (spin orbital, Nist)
    E0_cm_Mat = np.diagflat(E_cm)  # Make the diagonal energy matrix

    # States are [F,mF,J,mJ,I,mI,L,mL,S,mS]: couple (J,mJ) with (I,mI) into (F,mF) within a J manifold,
    # and (L,mL) with (S,mS) into (J,mJ) at fixed mI.
    w3cg1 = _cg_matrix(states, match=2, coupled=(2, 3, 4, 5), target=(0, 1))
    w3cg2 = _cg_matrix(states, match=5, coupled=(6, 7, 8, 9), target=(2, 3))
    w3cgprod = np.matmul(w3cg1, w3cg2)

    return [w3cgprod, E0_cm_Mat, E_HFSFlat, fourth, states]
