"""Zeeman/hyperfine energy fans versus magnetic field."""
import numpy as np

from .angular import lande_gf, lande_gi, lande_gj
from .constants import muBcm
from .hamiltonian import HZeeman, HZeeman_HFS, ZeemanHigh, ZeemanhighHFS, Zeemanlow
from .levels import stateprocess, stateprocess_HFS


def _constants_by_j(values, j_values):
    """Map one hyperfine constant per J (or one scalar for all J) onto the J values."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, len(j_values))
    if values.size != len(j_values):
        raise ValueError('HFS fan constants need one value per J, lowest J first')
    return dict(zip(j_values, values))


def FastZfan(States_in, Bfield, Eterm_=0, g_I=0, A_HFS=None, B_HFS=0):
    """Return [weak_field_energies, high_field_energies] in cm^-1 at Bfield (T).

    The approximations are diagonal, so each state's energy is evaluated directly
    instead of diagonalizing. Bfield may be a scalar or an array of fields; each
    returned entry then has the shape of Bfield.

    States_in comes from stateprocess or stateprocess_HFS. Without HFS, Eterm_
    sets the high-field term energy in cm^-1. With HFS, supply g_I from lande_gi
    and A_HFS/B_HFS in Hz (scalars or one value per J). The HFS high-field branch
    keeps J coupled and decouples J from I; it does not model the electronic
    Paschen-Back limit.
    """
    states = States_in[-1]
    zero_field = np.diag(States_in[1])
    is_hfs = len(states[0]) == 10
    if is_hfs:
        if A_HFS is None:
            raise ValueError('HFS fan approximations require A_HFS in Hz')
        j_values = sorted({state[2] for state in states})
        a_by_j = _constants_by_j(A_HFS, j_values)
        b_by_j = _constants_by_j(B_HFS, j_values)

    low, high = [], []
    for i, state in enumerate(states):
        if is_hfs:
            g_f = lande_gf(state[0], state[4], state[2], state[6], state[8], g_I)
            low.append(Zeemanlow(g_f, state[1], Bfield) + States_in[2][i] + zero_field[i])
            # The Paschen-Back expression already contains the diagonal HFS term,
            # so the F-dependent zero-field shift is not added again.
            high.append(ZeemanhighHFS(state, a_by_j[state[2]], g_I, Bfield, b_by_j[state[2]])
                        + zero_field[i])
        else:
            g_j = lande_gj(state[0], state[2], state[4])
            low.append(Zeemanlow(g_j, state[1], Bfield) + zero_field[i])
            high.append(ZeemanHigh(state[5], state[3], Bfield) + Eterm_)
    return [low, high]


def _exact_fan(zero_field, zeeman, fields):
    """Sorted eigenvalues of zero_field + B*muBcm*zeeman for every field, shape (states, fields).

    All fields are diagonalized in one batched call, about the mean energy so that
    small splittings on a large level energy keep full precision.
    """
    origin = float(np.mean(np.diag(zero_field)))
    shifted = zero_field - origin*np.eye(len(zero_field))
    stack = shifted[None, :, :] + fields[:, None, None]*(muBcm*zeeman)[None, :, :]
    return np.linalg.eigvalsh(stack).T + origin


def ZeemanFan(Inputdeck, flags=''):
    """Return energy fans versus the magnetic fields in Inputdeck['Bmag'].

    flags selects G (lower), E (upper), L (weak-field approximation), and
    H (high-field approximation). Return a dictionary with selected G/E,
    LowG/LowE and HighG/HighE arrays, shaped (states, field samples), in
    cm^-1. Exact rows follow energy rank; they do not track eigenstates
    through crossings. Non-HFS H requires the selected term's EtermG or
    EtermE, including zero. HFS H uses supplied A/B constants and keeps J."""
    fields = np.atleast_1d(np.asarray(Inputdeck['Bmag'], dtype=float))
    is_hfs = 'I_spin' in Inputdeck
    if not is_hfs and 'H' in flags:
        for flag, key in (('G', 'EtermG'), ('E', 'EtermE')):
            if flag in flags and Inputdeck.get(key) is None:
                raise ValueError(f'Non-HFS high-field {flag} fan requires {key} in cm^-1')
    g_I = lande_gi(Inputdeck['I_spin'], Inputdeck['mu_I']) if is_hfs else 0

    terms = (('G', 'ground', 'HFS_G', 'HFSB_G', 'EtermG'),
             ('E', 'excited', 'HFS_E', 'HFSB_E', 'EtermE'))
    output = {}
    for key, term, a_key, b_key, eterm_key in terms:
        if key not in flags:
            continue
        L, S, E = Inputdeck[f'l_{term}'], Inputdeck[f's_{term}'], Inputdeck[f'E_{term}']
        if is_hfs:
            constants = {'HFSA': Inputdeck[a_key], 'HFSB': Inputdeck.get(b_key, np.zeros_like(E))}
            level = stateprocess_HFS(L, S, Inputdeck['I_spin'], constants, g_I, E, 0)
            parts = HZeeman_HFS(level, g_I)
            zero_field, zeeman = parts['E0'] + np.diagflat(parts['EHF0']), parts['H_Z']
            approximation = dict(g_I=g_I, A_HFS=constants['HFSA'], B_HFS=constants['HFSB'])
        else:
            level = stateprocess(L, S, E, 0)
            zero_field, zeeman = HZeeman(level)
            eterm = Inputdeck.get(eterm_key)
            approximation = dict(Eterm_=0 if eterm is None else eterm)

        output[key] = _exact_fan(np.asarray(zero_field), np.asarray(zeeman), fields)
        if 'L' in flags or 'H' in flags:
            low, high = FastZfan(level, fields, **approximation)
            if 'L' in flags:
                output['Low' + key] = np.array(low, dtype=float)
            if 'H' in flags:
                output['High' + key] = np.array(high, dtype=float)
    return output
