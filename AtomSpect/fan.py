"""Zeeman/hyperfine energy fans versus magnetic field."""
import numpy as np

from .angular import lande_gf, lande_gi, lande_gj
from .constants import muBcm
from .hamiltonian import HZeeman, HZeeman_HFS, ZeemanHigh, ZeemanhighHFS, Zeemanlow
from .levels import stateprocess, stateprocess_HFS


#        stateprocces HFS return [w3cgprod,E0_cm_Mat,E_HFSFlat,E_low,bigstatelist]


def FastZfan(States_in, Bfield, Eterm_=0, g_I=0, A_HFS=None, B_HFS=0):
    
    # This is a supporting function which is for streamlining the diagonalization process which makes it much faster
    # Only the required calculation steps are taken for each magnetic field as the E0 and wavefunctions for zero-field don't change.
    # So make the wavefunctions once only before diagonalization.
    # Make a large list of [F,mF,J,mJ,I,mI,L,mL,S,mS] HFS or [J,mJ,L,mL,S,mS] no HFS
    """Return [weak_field_energies, high_field_energies] in cm^-1 at Bfield (T).

    States_in comes from stateprocess or stateprocess_HFS. Without HFS,
    Eterm_ sets the high-field term energy in cm^-1. With HFS, supply g_I
    from lande_gi and A_HFS/B_HFS in Hz (scalars or one value per J).
    The HFS high-field branch keeps J coupled and decouples J from I;
    it does not model the electronic Paschen-Back limit."""
    ZfansLow = []
    ZfansHigh = []
    is_hfs = len(States_in[-1][0]) == 10
    if is_hfs:
        if A_HFS is None:
            raise ValueError('HFS fan approximations require A_HFS in Hz')
        j_values = sorted({state[2] for state in States_in[-1]})
        def constants_by_j(values):
            values = np.asarray(values, dtype=float).reshape(-1)
            if values.size == 1:
                values = np.repeat(values, len(j_values))
            if values.size != len(j_values):
                raise ValueError('HFS fan constants need one value per J, lowest J first')
            return dict(zip(j_values, values))
        a_by_j = constants_by_j(A_HFS)
        b_by_j = constants_by_j(B_HFS)
    for i, states in enumerate(States_in[-1]):
        if not is_hfs:
            thislandegF = lande_gj(states[0], states[2], states[4])
            thisZeemanEnergyL = Zeemanlow(thislandegF, states[1], Bfield) + States_in[1][i][i]
            thisZeemanEnergyH = ZeemanHigh(states[5], states[3], Bfield) + Eterm_

        else:

            thislandegF = lande_gf(states[0], states[4], states[2], states[6], states[8], g_I)  # Calculate lande gf factor for the given transition
            thisZeemanEnergyL = Zeemanlow(thislandegF, states[1], Bfield) + States_in[2][i] + States_in[1][i][i]
            # Do not add the F-dependent zero-field HFS shift again: the
            # Paschen-Back expression already contains its diagonal HFS term.
            thisZeemanEnergyH = (ZeemanhighHFS(states, a_by_j[states[2]], g_I,
                                             Bfield, b_by_j[states[2]])
                                + States_in[1][i][i])

        ZfansLow.append(thisZeemanEnergyL)
        ZfansHigh.append(thisZeemanEnergyH)

    # print(Zfans[-1])
    return [ZfansLow, ZfansHigh]


def ZeemanFan(Inputdeck, flags=''):
    # Take the Inputdeck, generate states, then calculate energy for each value. Transpose and put it out.
    # Take the required values from the input deck.
    """Return energy fans versus the magnetic fields in Inputdeck['Bmag'].

    flags selects G (lower), E (upper), L (weak-field approximation), and
    H (high-field approximation). Return a dictionary with selected G/E,
    LowG/LowE and HighG/HighE arrays, shaped (states, field samples), in
    cm^-1. Exact rows follow energy rank; they do not track eigenstates
    through crossings. Non-HFS H requires the selected term's EtermG or
    EtermE, including zero. HFS H uses supplied A/B constants and keeps J."""

    S_ground = Inputdeck['s_ground']
    S_excited = Inputdeck['s_excited']
    L_ground = Inputdeck['l_ground']
    L_excited = Inputdeck['l_excited']
    E_ground = Inputdeck['E_ground']
    E_excited = Inputdeck['E_excited']
    Bfield_ext = Inputdeck['Bmag']
    # Bext_angle = Inputdeck['b_angle']
    EtermE = Inputdeck.get('EtermE')
    EtermG = Inputdeck.get('EtermG')
    if 'I_spin' not in Inputdeck and 'H' in flags:
        for flag, key in (('G', 'EtermG'), ('E', 'EtermE')):
            if flag in flags and Inputdeck.get(key) is None:
                raise ValueError(f'Non-HFS high-field {flag} fan requires {key} in cm^-1')
    # Generate empty lists for storing values in later. Ground, excited, low and exact. Look to add high approximation too.
    # GZ_fan = []
    # EZ_fan = []

    # Check if HFS is being included.
    if 'I_spin' in Inputdeck:
        HFSConstsG = {}
        HFSConstsG['HFSA'] = Inputdeck['HFS_G']
        HFSConstsE = {}
        HFSConstsE['HFSA'] = Inputdeck['HFS_E']

        if 'HFSB_G' in Inputdeck:  # Check if either of the B HFS constants are in the input deck, otherwise make a copy of the A coefficients as zeros.
            HFSConstsG['HFSB'] = Inputdeck['HFSB_G']
        else:
            HFSConstsG['HFSB'] = np.zeros_like(E_ground)

        if 'HFSB_E' in Inputdeck:  # Check if either of the B HFS constants are in the input deck, otherwise make a copy of the A coefficients as zeros.
            HFSConstsE['HFSB'] = Inputdeck['HFSB_E']
        else:
            HFSConstsE['HFSB'] = np.zeros_like(E_excited)

        I_spin = Inputdeck['I_spin']
        mu_I = Inputdeck['mu_I']
        thisg_I = lande_gi(I_spin, mu_I)

        output = {}
        if 'G' in flags:

            Ground_Level_HFS = stateprocess_HFS(L_ground, S_ground, I_spin, HFSConstsG, thisg_I, E_ground, 0)
            Ground_Zeeman_HFS = HZeeman_HFS(Ground_Level_HFS, thisg_I)

            EigG = []
            for Bvals in Bfield_ext:

                # tmat = np.asmatrix(Ground_Zeeman_HFS[0] +  Ground_Zeeman_HFS[2] +  Bvals*Ground_Zeeman_HFS[1]*muBcm)
                if 'PsiHFS' in Inputdeck:
                    tmat = np.asmatrix(Ground_Zeeman_HFS['E0'] + Ground_Zeeman_HFS['PsiHFS'] + Bvals*Ground_Zeeman_HFS['H_Z']*muBcm)
                else:
                    G_HFS0 = np.diagflat([x for x in Ground_Zeeman_HFS['EHF0']])
                    tmat = np.asmatrix(Ground_Zeeman_HFS['E0'] + G_HFS0 + Bvals*Ground_Zeeman_HFS['H_Z']*muBcm)

                EigG.append(np.linalg.eigh(tmat)[0])
            output['G'] = np.array(EigG).T

            if 'L' in flags:
                LowG = []
                for Bvals in Bfield_ext:
                    LowG.append(FastZfan(Ground_Level_HFS, Bvals, g_I=thisg_I,
                                        A_HFS=HFSConstsG['HFSA'], B_HFS=HFSConstsG['HFSB'])[0])

                output['LowG'] = np.array(LowG).T
            if 'H' in flags:
                HighG = []
                for Bvals in Bfield_ext:
                    HighG.append(FastZfan(Ground_Level_HFS, Bvals, g_I=thisg_I,
                                         A_HFS=HFSConstsG['HFSA'], B_HFS=HFSConstsG['HFSB'])[1])

                output['HighG'] = np.array(HighG).T
        if 'E' in flags:
            Excited_Level_HFS = stateprocess_HFS(L_excited, S_excited,  I_spin, HFSConstsE, thisg_I, E_excited, 0)
            Excited_Zeeman_HFS = HZeeman_HFS(Excited_Level_HFS, thisg_I)

            EigE = []
            for Bvals in Bfield_ext:

                if 'PsiHFS' in Inputdeck:
                    tmat = np.asmatrix(Excited_Zeeman_HFS['E0'] + Excited_Zeeman_HFS['PsiHFS'] + Bvals*Excited_Zeeman_HFS['H_Z']*muBcm)
                else:
                    E_HFS0 = np.diagflat([x for x in Excited_Zeeman_HFS['EHF0']])
                    tmat = np.asmatrix(Excited_Zeeman_HFS['E0'] + E_HFS0 + Bvals*Excited_Zeeman_HFS['H_Z']*muBcm)

                EigE.append(np.linalg.eigh(tmat)[0])
            output['E'] = np.array(EigE).T
            if 'L' in flags:
                LowE = []
                for Bvals in Bfield_ext:
                    LowE.append(FastZfan(Excited_Level_HFS, Bvals, g_I=thisg_I,
                                        A_HFS=HFSConstsE['HFSA'], B_HFS=HFSConstsE['HFSB'])[0])

                output['LowE'] = np.array(LowE).T
            if 'H' in flags:
                HighE = []
                for Bvals in Bfield_ext:
                    HighE.append(FastZfan(Excited_Level_HFS, Bvals, g_I=thisg_I,
                                         A_HFS=HFSConstsE['HFSA'], B_HFS=HFSConstsE['HFSB'])[1])

                output['HighE'] = np.array(HighE).T

    else:
        output = {}
        if 'G' in flags:
            Ground_Level = stateprocess(L_ground, S_ground, E_ground, 0)
            Ground_Zeeman = HZeeman(Ground_Level)

            EigG = []
            for Bvals in Bfield_ext:

                tmat = np.asmatrix(Ground_Zeeman[0] + Bvals*Ground_Zeeman[1]*muBcm)

                EigG.append(np.linalg.eigh(tmat)[0])
            output['G'] = np.array(EigG).T

            if 'L' in flags:
                LowG = []
                for Bvals in Bfield_ext:
                    LowG.append(FastZfan(Ground_Level, Bvals)[0])

                output['LowG'] = np.array(LowG).T
            if 'H' in flags:
                HighG = []
                for Bvals in Bfield_ext:
                    HighG.append(FastZfan(Ground_Level, Bvals,Eterm_=EtermG)[1])

                output['HighG'] = np.array(HighG).T
        if 'E' in flags:
            Excited_Level = stateprocess(L_excited, S_excited, E_excited, 0)
            Excited_Zeeman = HZeeman(Excited_Level)

            EigE = []
            for Bvals in Bfield_ext:

                tmat = np.asmatrix(Excited_Zeeman[0] + Bvals*Excited_Zeeman[1]*muBcm)

                EigE.append(np.linalg.eigh(tmat)[0])
            output['E'] = np.array(EigE).T

            if 'L' in flags:
                LowE = []
                for Bvals in Bfield_ext:
                    LowE.append(FastZfan(Excited_Level, Bvals)[0])

                output['LowE'] = np.array(LowE).T
            if 'H' in flags:
                HighE = []
                for Bvals in Bfield_ext:
                    HighE.append(FastZfan(Excited_Level, Bvals,Eterm_=EtermE)[1])

                output['HighE'] = np.array(HighE).T
    return output
