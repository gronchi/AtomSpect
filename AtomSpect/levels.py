"""Construction of fine-structure and hyperfine state lists."""
import numpy as np

from .angular import CGw3, lande_gf, lande_gj, make_Jmj
from .hamiltonian import Zeemanlow
from .hfs import Delta_E_HFS


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
    '''
    Takes input state values and magnetic field to expand the state space. 
    L and S are used to calculate possible J values. all possible mJ values are also created. 
    E_level is an np.array and must match the length of possible J values. Since the values are pulled from NIST, this will likely happen.


    Parameters
    ----------
    L_state : Int or half int
        Orbital angular quantum number. S=0, P=1, D=2, F=3, etc
    S_state : Int or half int
        Spin quantum number, n = 2s+1 for multiplicity, so ^2S_J has (2*s +1) =2, so s=0.5, L=0
    E_level : np.array
        Energy in cm^-1, Lowest J value first.
    thisB : 
        Magnetic field in T

    Returns
    -------
    list
       [w3mat, Jmjs, LS,E_low, E_cm,E_SO_cm_Mat,sortedstates]
    w3mat is the state array of w3 coefficients corresponding to CG coefficients.
    Jmjs are the possible Jmjs #Deprecated
    LS is the L,mL,S,mS states. #Deprecated
    E_low is a list of energies corresponding to the lowfield zeeman approx
    E_cm is the sorted energies in cm^-1
    E_SO_cm_Mat is the sorted (diagonal) matrix of energies with repeated values for degeneracy of mJ
    sortedstates is [J,mJ,L,mL,S,mS] states corresponding to the sorted energies in the matrix. Off diagonals are excluded to this point.

    '''

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

    HFS_A_In = HFS_Const['HFSA']
    try:
        HFSConst = [[HFS_A_In[x]]*int(2*Jvals[x] + 1)*int(2*I_state + 1) for x in range(len(E_level))]
        HFS_Flat = [item for row in HFSConst for item in row]
    except:
        # This handles the case of single length HFS inputs
        HFSConst = [[HFS_A_In]*int(2*Jvals[x] + 1)*int(2*I_state + 1) for x in range(len(E_level))]
        HFS_Flat = [item for row in HFSConst for item in row]

        # HFS_Flat = HFSConst
    # print(HFSConst)

    if 'HFSB' in HFS_Const:
        try:
            HFSConstB = [[HFS_Const['HFSB'][x]]*int(2*Jvals[x] + 1)*int(2*I_state + 1) for x in range(len(E_level))]
            HFS_FlatB = [item for row in HFSConstB for item in row]
        except:
            HFSConstB = [list(HFS_Const['HFSB'])*int(2*Jvals[x] + 1)*int(2*I_state + 1) for x in range(len(E_level))]
            HFS_FlatB = [item for row in HFSConstB for item in row]
    else:
        HFS_FlatB = np.zeros_like(E0_cm_flat)
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
