"""Construction of fine-structure and hyperfine state lists."""
import numpy as np

from .angular import CGw3, lande_gf, lande_gj, make_Jmj
from .hamiltonian import Zeemanlow
from .hfs import Delta_E_HFS


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
    E_cm0_Mat = np.diagflat(E_cm0_flat)
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
        sortedstates = [x[-1] for x in sortedlist]
        E_cm_flat = [x[1] for x in sortedlist]  # This is the sorted energies, and is in cm^-1
        # Jmjs = [x[2] for x in sortedlist]
        # LS = [x[3] for x in sortedlist]
        E_SO_cm_Mat = np.diagflat([item for item in E_cm_flat])  # Despite the name, these energies are actually in cm-1

    # We now use wigner 3j to get an equivalent decoupling as per how CG coefficients work.
    # The result is a 2d where the columns are the |J,mJ> states and the rows determine what the corresponding |L,mL,S,mS> state was.
    # Eventually would like a way to convert this data to a more human readable form
        dim = len(Jmjs0)
        w3mat = np.zeros((dim, dim))
        # [J,mJ,L,mL,S,mS]
        for i, states1 in enumerate(sortedstates):

            for j, states2 in enumerate(sortedstates):
                # This conditional is saying that J != J', as otherwise the different J levels will cause duplicate CG's.
                if states1[2] != states2[2]:
                    CG1 = 0
                else:
                    # The CG coefficient couples J,mJ to L,mL, S mS
                    CG1 = CGw3(states2[2], states2[3], states2[4], states2[5], states1[0], states1[1])
                w3mat[i, j] = CG1
        return [w3mat, E_SO_cm_Mat, LowE, E_cm_flat, sortedstates]

    else:
        dim = len(Jmjs0)
        w3mat = np.zeros((dim, dim))
        # [J,mJ,L,mL,S,mS]
        for i, states1 in enumerate(bigstatelist):

            for j, states2 in enumerate(bigstatelist):
                # This conditional is saying that J != J', as otherwise the different J levels will cause duplicate CG's.
                if states1[2] != states2[2]:
                    CG1 = 0
                else:
                    # The CG coefficient couples J,mJ to L,mL, S mS
                    CG1 = CGw3(states2[2], states2[3], states2[4], states2[5], states1[0], states1[1])
                w3mat[i, j] = CG1
        return [w3mat, E_cm0_Mat, LowE, E_cm0_flat, bigstatelist]


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
            # thislandegF = lande_gf(bigstatelist[i][0],I_state, bigstatelist[i][2], L_state,S_state,g_I) #Calculate lande gf factor for the given transition
            thislandegF = lande_gf(bigstatelist[i][0], bigstatelist[i][4], bigstatelist[i][2], bigstatelist[i][6], bigstatelist[i][8], g_I)  # Calculate lande gf factor for the given transition

            # Lowfield zeeman approximation with hfs, which should be mf and gf.
            thisZeemanEnergy = Zeemanlow(thislandegF, bigstatelist[i][1], thisB) + x + Delta_E_HFS(I_state, bigstatelist[i][2], bigstatelist[i][0], HFS_Flat[i], HFS_FlatB[i])

            # thisZeemanEnergy = Zeemanlow(thislandegJ,bigstatelist[i][3],thisB) + muBcm*g_I*thisB*bigstatelist[i][5] + x + Delta_E_HFS(I_state,bigstatelist[i][2],bigstatelist[i][0],HFS_Flat[i],HFS_FlatB[i])
            biglist.append([thisZeemanEnergy, x, bigstatelist[i], HFS_Flat[i], HFS_FlatB[i]])
        # Sort the results and pull sorted values
        sortedlist = sorted(biglist)
        sortedstates = [x[2] for x in sortedlist]
        sortedHFC = [x[3] for x in sortedlist]  # Sorted hyperfine constants
        sortedHFCB = [x[4] for x in sortedlist]  # Sorted hyperfine constants

        E_cm = [x[1] for x in sortedlist]
        E_low = [x[0] for x in sortedlist]

        # Calculate Hyperfine energies with sorted states and values.
        E_HFSFlat = [Delta_E_HFS(states1[4], states1[2], states1[0], sortedHFC[i], sortedHFCB[i]) for i, states1 in enumerate(sortedstates)]  # Hyperfine + E0 (spin orbital, Nist)

        # E_HFSFlat = [E_cm[i] + Delta_E_HFS(states1[4],states1[2],states1[0], sortedHFC[i], sortedHFCB[i])    for i,states1 in enumerate(sortedstates)] #Hyperfine + E0 (spin orbital, Nist)
        # E_HFSA = [Delta_E_HFS(states1[4],states1[2],states1[0], sortedHFC[i],0)    for i,states1 in enumerate(sortedstates)] #Just Hyperfine energy
        # E_HFSB = [Delta_E_HFS(states1[4],states1[2],states1[0], 0, B=sortedHFCB[i])    for i,states1 in enumerate(sortedstates)] #Just Hyperfine energy

        # E0_cm_Mat = np.diagflat([item for item in E_HFSFlat]) #Make the diagonal energy matrix
        E0_cm_Mat = np.diagflat([item for item in E_cm])  # Make the diagonal energy matrix

        # We now use wigner 3j to get an equivalent decoupling as per how CG coefficients work.
        # There are two coefficients as we go from |F,mF> -> |J,mJ>|I,mI> then another set for |J,mJ>|I,mI> -> |L,mL>|S,mS>|I,mI>. Keeping track of constants is slightly tricky.
        # There can be issues with overla
        # Eventually would like a way to convert this data to a more human readable form
        dimhfs = len(Fmfs0)
        w3cg1 = np.zeros((dimhfs, dimhfs))
        w3cg2 = np.zeros((dimhfs, dimhfs))
        # Done in a single set of iterations. The check for states1[2] != states2[2] is saying that J != J' to calculate, otherwise it's zero.
        for i, states1 in enumerate(sortedstates):

            for j, states2 in enumerate(sortedstates):
                # This conditional is saying that J != J', as otherwise the different J levels will cause duplicate CG's.
                if states1[2] != states2[2]:
                    CG1 = 0
                else:
                    # Make a large list of [F,mF,J,mJ,I,mI,L,mL,S,mS]

                    # The CG coefficient couples F,mF to J,mJ;I,mI for CG1
                    CG1 = CGw3(states2[2], states2[3], states2[4], states2[5], states1[0], states1[1])
                # This conditional ensures we don't allow nuclear spin to flip or change between states. states[5] is mI.
                if states1[5] != states2[5]:
                    CG2 = 0
                    # CG2 = CGw3(states2[6],states2[7],states2[8],states2[9],states1[2],states1[3])
                    # w3cg2[i,j] = CG2

                else:
                    CG2 = CGw3(states2[6], states2[7], states2[8], states2[9], states1[2], states1[3])

                    w3cg2[i, j] = CG2
                w3cg1[i, j] = CG1

        w3cgprod = np.matmul(w3cg1, w3cg2)

        return [w3cgprod, E0_cm_Mat, E_HFSFlat, E_low, sortedstates]
    else:

        # E_cm = [x[1] for x in biglist]

        # Calculate Hyperfine energies with sorted states and values.
        E_HFSFlat = [Delta_E_HFS(states1[4], states1[2], states1[0], HFS_Flat[i], HFS_FlatB[i]) for i, states1 in enumerate(bigstatelist)]  # Hyperfine + E0 (spin orbital, Nist)

        #Holdovers from work with HFS energy contributions. Left for future possible use.
        # E0_HFSFlat = [E_cm[i] + Delta_E_HFS(states1[4],states1[2],states1[0], HFS_Flat[i], HFS_FlatB[i])    for i,states1 in enumerate(bigstatelist)] #Hyperfine + E0 (spin orbital, Nist)
        # E_HFSA = [Delta_E_HFS(states1[4],states1[2],states1[0], sortedHFC[i],0)    for i,states1 in enumerate(sortedstates)] #Just Hyperfine energy
        # E_HFSB = [Delta_E_HFS(states1[4],states1[2],states1[0], 0, B=sortedHFCB[i])    for i,states1 in enumerate(sortedstates)] #Just Hyperfine energy

        E0_cm_Mat = np.diagflat([item for item in E0_cm_flat])  # Make the diagonal energy matrix

        # We now use wigner 3j to get an equivalent decoupling as per how CG coefficients work.
        # There are two coefficients as we go from |F,mF> -> |J,mJ>|I,mI> then another set for |J,mJ>|I,mI> -> |L,mL>|S,mS>|I,mI>. Keeping track of constants is slightly tricky.
        # There can be issues with overla
        # Eventually would like a way to convert this data to a more human readable form
        dimhfs = len(Fmfs0)
        w3cg1 = np.zeros((dimhfs, dimhfs))
        w3cg2 = np.zeros((dimhfs, dimhfs))
        # Done in a single set of iterations. The check for states1[2] != states2[2] is saying that J != J' to calculate, otherwise it's zero.
        for i, states1 in enumerate(bigstatelist):

            for j, states2 in enumerate(bigstatelist):
                # This conditional is saying that J != J', as otherwise the different J levels will cause duplicate CG's.
                if states1[2] != states2[2]:
                    CG1 = 0

                else:
                    # Make a large list of [F,mF,J,mJ,I,mI,L,mL,S,mS]

                    # The CG coefficient couples F,mF to J,mJ;I,mI for CG1
                    CG1 = CGw3(states2[2], states2[3], states2[4], states2[5], states1[0], states1[1])
                # This conditional ensures we don't allow nuclear spin to flip or change between states. states[5] is mI.
                if states1[5] != states2[5]:
                    CG2 = 0
                    # CG2 = CGw3(states2[6],states2[7],states2[8],states2[9],states1[2],states1[3])
                    # w3cg2[i,j] = CG2

                else:
                    CG2 = CGw3(states2[6], states2[7], states2[8], states2[9], states1[2], states1[3])

                    w3cg2[i, j] = CG2
                w3cg1[i, j] = CG1

        w3cgprod = np.matmul(w3cg1, w3cg2)
        return [w3cgprod, E0_cm_Mat, E_HFSFlat, HFS_Flat, bigstatelist]
