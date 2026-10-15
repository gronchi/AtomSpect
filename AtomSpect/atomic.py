"""Phase-consistent, externally normalized Zeeman components (Models A/B/C).

This API implements the E1 core of the revised guide without evaluating
radial wavefunctions. Old relative-spectrum entry points are independent.
"""
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .atomic_data import (AtomicConstants, AtomicLevel, HyperfineData, LSTerm,
                          MissingAtomicData, RadiativeReference, _cg, _finite,
                          _rank_allowed, _two, _w3, _w6, e1_rate)

__all__ = ['AtomicModel', 'AtomicComponents', 'build_model_a', 'build_model_b',
           'build_model_c', 'calculate_components', 'calculate_field_grid',
           'add_hyperfine', 'add_dc_stark', 'rephase_model', 'components_from_inputdeck',
           'synthesize_spectrum', 'frequency_profile_to_wavelength', 'isolated_multipole_rates']


@dataclass
class AtomicModel:
    """Compiled fixed-basis operators; H0 is cm^-1, T0 dimensionless, Dq au.

basis records carry level IDs, parity and exact two_Mtotal. Model builders
validate missing links before compilation. Metadata retains original inputs.
"""
    model_id: str
    levels: tuple
    basis: tuple
    H0: np.ndarray
    T0: np.ndarray
    Dq: dict
    known_pairs: np.ndarray
    constants: AtomicConstants = field(default_factory=AtomicConstants)
    validity_flags: tuple = ()
    metadata: dict = field(default_factory=dict)

    def validate(self):
        """Verify Hermiticity, tensor adjoints, shapes and axial symmetry."""
        n = len(self.basis)
        for name, matrix in [('H0', self.H0), ('T0', self.T0)]:
            if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)):
                raise ValueError(f'Invalid {name} matrix')
            if not np.allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-12):
                raise ValueError(f'{name} must be Hermitian')
        m = np.array([b['two_Mtotal'] for b in self.basis])
        parity = np.array([b['parity'] for b in self.basis])
        same_block = (m[:, None] == m) & (parity[:, None] == parity)
        for matrix in (self.H0, self.T0):
            if np.any(np.abs(matrix[~same_block]) > 1e-12):
                raise ValueError('Hamiltonian operators must preserve total M and parity')
        if set(self.Dq) != {-1, 0, 1} or self.known_pairs.shape != (n, n):
            raise ValueError('Invalid dipole components or requested-pair mask')
        for q, matrix in self.Dq.items():
            if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)):
                raise ValueError('Invalid dipole matrix')
            if not np.allclose(matrix.conj().T, (-1)**q*self.Dq[-q], rtol=1e-12, atol=1e-12):
                raise ValueError('Dipole matrices violate Dq adjoint = (-1)^q D(-q)')
            allowed = (m[:, None]-m == 2*q) & (parity[:, None] != parity)
            if np.any(np.abs(matrix[~allowed]) > 1e-12):
                raise ValueError('Dipole matrices violate axial E1 symmetry')
        return self


def _levels(levels):
    values = tuple(x if isinstance(x, AtomicLevel) else AtomicLevel(**x) for x in levels)
    if not values or len({x.level_id for x in values}) != len(values):
        raise ValueError('A nonempty list of uniquely identified levels is required')
    if len({x.energy_reference_id for x in values}) != 1:
        raise ValueError('Every level must share one energy reference')
    if len({x.species_id for x in values}) != 1 or len({x.two_J % 2 for x in values}) != 1:
        raise ValueError('Levels must describe one species with compatible angular momenta')
    if len({x.isotope_id for x in values if x.isotope_id is not None}) > 1:
        raise ValueError('Compute different isotopes separately and combine intensities')
    return values


def _basis(levels):
    return tuple(dict(level_id=x.level_id, two_J=x.two_J, two_Mtotal=m,
                      two_mJ=m, parity=x.parity)
                 for x in levels for m in range(-x.two_J, x.two_J+1, 2))


def _base(levels):
    basis = _basis(levels)
    by_id = {x.level_id: x for x in levels}
    H0 = np.diag([by_id[b['level_id']].energy_cm for b in basis]).astype(complex)
    return basis, H0


def _complete_reduced(levels, entries, electric):
    """Require allowed links explicitly; only symmetry-forbidden links are zero."""
    by_id = {x.level_id: x for x in levels}
    result = {}
    for key, value in entries.items():
        if not isinstance(key, tuple) or len(key) != 2 or any(x not in by_id for x in key):
            raise ValueError(f'Invalid reduced-matrix link: {key}')
        z = complex(value)
        if not np.isfinite(z):
            raise ValueError(f'Nonfinite reduced element: {key}')
        a, b = (by_id[x] for x in key)
        allowed = _rank_allowed(a.two_J, b.two_J) and ((a.parity != b.parity) if electric else (a.parity == b.parity))
        if not allowed and z != 0:
            raise ValueError(f'Nonzero symmetry-forbidden reduced element: {key}')
        reverse = (-1)**((b.two_J-a.two_J)//2)*z.conjugate()
        for link, amplitude in ((key, z), ((key[1], key[0]), reverse)):
            if link in result and not np.isclose(result[link], amplitude, rtol=1e-12, atol=1e-14):
                raise ValueError(f'Inconsistent Hermitian reduced elements: {link}')
            result[link] = amplitude
    for a in levels:
        for b in levels:
            allowed = _rank_allowed(a.two_J, b.two_J) and ((a.parity != b.parity) if electric else (a.parity == b.parity))
            key = (a.level_id, b.level_id)
            if allowed and key not in result:
                kind = 'E1' if electric else 'magnetic'
                raise MissingAtomicData(f'Unknown required {kind} reduced element {key}; supply an explicit zero if known')
            if not allowed:
                result[key] = 0j
    return result


def _tensor(basis, reduced, q):
    matrix = np.zeros((len(basis), len(basis)), complex)
    for i, a in enumerate(basis):
        for j, b in enumerate(basis):
            if a['two_Mtotal']-b['two_Mtotal'] != 2*q:
                continue
            value = reduced.get((a['level_id'], b['level_id']), 0j)
            if value:
                matrix[i, j] = ((-1)**((a['two_J']-a['two_Mtotal'])//2)
                                * _w3(a['two_J'], 2, b['two_J'], -a['two_Mtotal'], 2*q, b['two_Mtotal'])*value)
    return matrix


def _flags(extra=()):
    return tuple(dict.fromkeys(('STARK_NOT_INCLUDED', 'DIAMAGNETISM_NOT_INCLUDED',
                               'TRUNCATION_UNASSESSED', 'HYPERFINE_UNASSESSED',
                               'PARTIAL_DECAY_SUM', 'UNCERTAINTY_UNASSESSED') + tuple(extra)))


def build_model_a(levels, references, *, constants=None, isolation_assessed=False):
    """Independent isolated-J E1 lines, using supplied gJ and individual data.

Only requested links are emitted. No coherent combination of independent
line amplitudes or unknown interlevel magnetic couplings is inferred.
"""
    levels = _levels(levels)
    k = constants or AtomicConstants()
    basis, H0 = _base(levels)
    by_id = {x.level_id: x for x in levels}
    if any(x.g_J is None for x in levels):
        raise MissingAtomicData('Model A requires g_J for every level')
    T0 = np.diag([by_id[b['level_id']].g_J*b['two_mJ']/2 for b in basis]).astype(complex)
    reduced, originals = {}, []
    for reference in references:
        r = reference if isinstance(reference, RadiativeReference) else RadiativeReference(**reference)
        if r.upper_level_id not in by_id or r.lower_level_id not in by_id:
            raise ValueError('Unknown radiative level ID')
        u, l = by_id[r.upper_level_id], by_id[r.lower_level_id]
        if u.parity == l.parity or not _rank_allowed(u.two_J, l.two_J):
            raise ValueError('Reference violates E1 parity or angular selection rules')
        key = (u.level_id, l.level_id)
        amplitude = np.sqrt(r.strength_au(u, l, k))
        if key in reduced:
            raise ValueError('Duplicate independent line reference')
        reduced[key] = amplitude
        reduced[key[::-1]] = (-1)**((l.two_J-u.two_J)//2)*amplitude
        originals.append(asdict(r))
    if not originals:
        raise MissingAtomicData('Model A requires at least one identified E1 line')
    known = np.array([[(a['level_id'], b['level_id']) in reduced for b in basis] for a in basis])
    model = AtomicModel('A', levels, basis, H0, T0, {q: _tensor(basis, reduced, q) for q in (-1, 0, 1)}, known, k,
                        _flags(() if isolation_assessed else ('ISOLATION_UNASSESSED',)),
                        dict(references=originals, phase_known=False, independent_lines=True,
                             constants=asdict(k), source_ids=sorted({x.source_id for x in levels} | {r['source_id'] for r in originals})))
    return model.validate()


def _ls_transform(term, basis):
    """Rows are coupled |J,M>; columns are orbital-first |mL,mS>."""
    uncoupled = [(ml, ms) for ml in range(-2*term.L, 2*term.L+1, 2)
                 for ms in range(-term.two_S, term.two_S+1, 2)]
    U = np.array([[_cg(2*term.L, ml, term.two_S, ms, b['two_J'], b['two_mJ'])
                   for ml, ms in uncoupled] for b in basis])
    if not np.allclose(U @ U.T, np.eye(len(basis)), atol=1e-12):
        raise ValueError('Incomplete or inconsistent LS coupling basis')
    return U, uncoupled


def build_model_b(lower_term, upper_term, references=(), *, orbital_strength_au=None,
                  constants=None, consistency_rtol=1e-6):
    """Complete isolated LS terms with one norm per orbital channel.

Reference level IDs are term_id:J<two_J>. Each independent branch is
checked against the common norm; inconsistent data are rejected rather than
calibrating separate branches. consistency_rtol is a caller-adjustable
numerical tolerance, augmented by the supplied uncertainty intervals.
"""
    lower = lower_term if isinstance(lower_term, LSTerm) else LSTerm(**lower_term)
    upper = upper_term if isinstance(upper_term, LSTerm) else LSTerm(**upper_term)
    if lower.term_id == upper.term_id or lower.two_S != upper.two_S:
        raise ValueError('Model B requires distinct term IDs and equal electronic spin')
    if lower.parity == upper.parity or not _rank_allowed(2*upper.L, 2*lower.L):
        raise ValueError('Model B requires an allowed opposite-parity orbital E1 channel')
    _finite(consistency_rtol, 'consistency_rtol', minimum=0)
    k = constants or AtomicConstants()
    lower_levels, upper_levels = lower.levels(), upper.levels()
    levels = _levels(lower_levels + upper_levels)
    by_id = {x.level_id: x for x in levels}
    candidates, originals = [], []
    for reference in references:
        r = reference if isinstance(reference, RadiativeReference) else RadiativeReference(**reference)
        if r.upper_level_id not in {x.level_id for x in upper_levels} or r.lower_level_id not in {x.level_id for x in lower_levels}:
            raise ValueError('Reference must link the selected upper and lower terms')
        u, l = by_id[r.upper_level_id], by_id[r.lower_level_id]
        factor = (u.two_J+1)*(l.two_J+1)*_w6(2*upper.L, u.two_J, upper.two_S, l.two_J, 2*lower.L, 2)**2
        if factor <= 0:
            raise ValueError('A forbidden LS branch cannot normalize this channel')
        norm = r.strength_au(u, l, k)/factor
        candidates.append((norm, norm*r.relative_uncertainty))
        originals.append(asdict(r))
    if orbital_strength_au is None:
        if not candidates or candidates[0][0] <= 0:
            raise MissingAtomicData('Model B needs a positive reference branch or an explicit orbital_strength_au')
        norm, error = candidates[0]
    else:
        norm, error = _finite(orbital_strength_au, 'orbital_strength_au', minimum=0), 0.
    for other, uncertainty in candidates:
        if abs(other-norm) > consistency_rtol*max(other, norm) + error + uncertainty:
            raise ValueError('Reference branches disagree with the strict-LS common orbital norm')
    basis, H0 = _base(levels)
    nlower = sum(x.two_J+1 for x in lower_levels)
    T0 = np.zeros_like(H0)
    for term, indices in ((lower, slice(0, nlower)), (upper, slice(nlower, len(basis)))):
        U, unc = _ls_transform(term, basis[indices])
        projections = np.array([(ml+k.g_s*ms)/2 for ml, ms in unc])
        T0[indices, indices] = (U*projections) @ U.T
    reduced = {}
    for u in upper_levels:
        for l in lower_levels:
            if _rank_allowed(u.two_J, l.two_J):
                phase = (-1)**((2*upper.L + upper.two_S + l.two_J + 2)//2)
                value = phase*np.sqrt((u.two_J+1)*(l.two_J+1)*norm)*_w6(2*upper.L, u.two_J, upper.two_S, l.two_J, 2*lower.L, 2)
            else:
                value = 0.
            reduced[(u.level_id, l.level_id)] = value
            reduced[(l.level_id, u.level_id)] = (-1)**((l.two_J-u.two_J)//2)*value
    known = np.array([[a['parity'] != b['parity'] for b in basis] for a in basis])
    metadata = dict(lower_term=asdict(lower), upper_term=asdict(upper), references=originals,
                    orbital_strength_au=norm, branch_norms_au=[x for x, _ in candidates],
                    consistency_rtol=consistency_rtol, phase_convention_id='orbital-first Condon-Shortley',
                    constants=asdict(k), source_ids=sorted({x.source_id for x in levels} | {r['source_id'] for r in originals}))
    return AtomicModel('B', levels, basis, H0, T0, {q: _tensor(basis, reduced, q) for q in (-1, 0, 1)},
                       known, k, _flags(('LS_ASSUMED', 'PHASES_ASSUMED_FROM_MODEL')), metadata).validate()


def build_model_c(levels, reduced_e1, reduced_magnetic, *, phase_convention_id,
                  matrix_dataset_id, source_ids=(), dipole_units='au', magnetic_units='dimensionless', constants=None):
    """General electronic mixing with phase-consistent external reduced matrices.

All symmetry-allowed retained links require explicit entries, including
known zeros. Diagonal magnetic entries may be supplied through level gJ.
Dipoles are au or SI (C m); physical magnetic moments in J/T are converted
to T=-mu/muB. No off-diagonal information is inferred from unsigned rates.
"""
    levels = _levels(levels)
    if not phase_convention_id or not matrix_dataset_id:
        raise MissingAtomicData('Model C requires phase_convention_id and matrix_dataset_id')
    k = constants or AtomicConstants()
    if dipole_units not in ('au', 'SI') or magnetic_units not in ('dimensionless', 'J/T'):
        raise ValueError('Unsupported reduced-operator unit convention')
    r = {key: complex(value)/(np.sqrt(k.strength_unit_SI) if dipole_units == 'SI' else 1.) for key, value in reduced_e1.items()}
    muB = k.mu_B_cm*100*k.h*k.c
    t = {key: complex(value)*(-1/muB if magnetic_units == 'J/T' else 1.) for key, value in reduced_magnetic.items()}
    for level in levels:
        if level.g_J is not None and level.two_J:
            J = level.two_J/2
            value = level.g_J*np.sqrt(J*(J+1)*(2*J+1))
            key = (level.level_id, level.level_id)
            if key in t and not np.isclose(t[key], value, rtol=1e-10, atol=1e-12):
                raise ValueError('Diagonal magnetic reduced element disagrees with supplied g_J')
            t[key] = value
    r = _complete_reduced(levels, r, True)
    t = _complete_reduced(levels, t, False)
    basis, H0 = _base(levels)
    known = np.array([[a['parity'] != b['parity'] for b in basis] for a in basis])
    metadata = dict(phase_convention_id=phase_convention_id, matrix_dataset_id=matrix_dataset_id,
                    source_ids=list(dict.fromkeys([x.source_id for x in levels] + list(source_ids))),
                    original_reduced_e1=dict(reduced_e1), original_reduced_magnetic=dict(reduced_magnetic),
                    dipole_units=dipole_units, magnetic_units=magnetic_units, constants=asdict(k))
    return AtomicModel('C', levels, basis, H0, _tensor(basis, t, 0),
                       {q: _tensor(basis, r, q) for q in (-1, 0, 1)}, known, k, _flags(), metadata).validate()


def add_dc_stark(model, electric_field_v_cm=0., polarizabilities=None):
    """Return a model with axial dc scalar/tensor Stark energy shifts.

    polarizabilities maps every level ID to (alpha0/h, alpha2/h) in
    Hz/(V/cm)^2. The field is along the quantization axis (parallel to B).
    The tensor is diagonal in electronic J,mJ and can mix F when HFS is
    present. Cross-J polarizabilities are omitted. Supply absolute level
    polarizabilities, rather than excited-minus-ground differences.
    Zero field returns the original model without requiring data. Apply
    once to a model; repeated application would double-count its shifts.
    """
    from .stark import _electric_field, _stark_shifts
    electric = _electric_field(electric_field_v_cm)
    if electric == 0:
        return model
    model.validate()
    if 'dc_stark' in model.metadata:
        raise ValueError('DC Stark shifts have already been added to this model')
    if polarizabilities is None or set(polarizabilities) != {x.level_id for x in model.levels}:
        raise MissingAtomicData('DC Stark polarizabilities must cover every retained level')
    records = {}
    for level in model.levels:
        values = np.asarray(polarizabilities[level.level_id], float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError('Each polarizability record must contain finite (alpha0/h, alpha2/h)')
        records[level.level_id] = tuple(values.tolist())
    js = np.array([b['two_J']/2 for b in model.basis])
    ms = np.array([b['two_mJ']/2 for b in model.basis])
    a0, a2 = np.array([records[b['level_id']] for b in model.basis]).T
    shift = _stark_shifts(js, ms, electric, a0, a2, model.constants.c)
    metadata = dict(model.metadata, dc_stark=dict(electric_field_v_cm=electric,
                    polarizabilities_Hz_per_V_cm_squared=records, axis='z'))
    flags = tuple(x for x in model.validity_flags if x != 'STARK_NOT_INCLUDED')
    flags += ('STARK_CROSS_J_NOT_INCLUDED',)
    return replace(model, H0=model.H0+np.diag(shift), metadata=metadata,
                   validity_flags=flags).validate()


def add_hyperfine(model, data):
    """Enlarge an electronic model with one isotope and diagonal-J HFS.

Uses |J,mJ;I,mI>, HFS A/B in Hz, and the physical nuclear-moment sign.
Off-diagonal electronic hyperfine tensors are explicitly flagged as omitted.
"""
    model.validate()
    hfs = data if isinstance(data, HyperfineData) else HyperfineData(**data)
    I2 = _two(hfs.two_I, 'two_I')
    mu = _finite(hfs.mu_I, 'mu_I')
    if not hfs.isotope_id or not hfs.source_id:
        raise ValueError('Hyperfine data require isotope and source IDs')
    if any(r.get('two_I') is not None and r['two_I'] != I2 for r in model.metadata.get('references', [])):
        raise ValueError('Reference nuclear spin disagrees with the requested isotope')
    if any('two_mI' in b for b in model.basis):
        raise ValueError('This model already contains nuclear spin')
    if I2 == 0 and mu != 0:
        raise ValueError('Spin-zero nuclei require zero magnetic moment')
    if set(hfs.constants_Hz) != {x.level_id for x in model.levels}:
        raise MissingAtomicData('HFS constants must explicitly cover every retained level')
    nuclear = list(range(-I2, I2+1, 2))
    basis = tuple(dict(b, two_mI=mi, two_Mtotal=b['two_mJ']+mi) for b in model.basis for mi in nuclear)
    H0 = np.kron(model.H0, np.eye(I2+1))
    for level in model.levels:
        values = hfs.constants_Hz[level.level_id]
        if np.shape(values) != (2,):
            raise ValueError('Each HFS record must contain (A_Hz, B_Hz)')
        Ahz, Bhz = (_finite(x, 'HFS constant') for x in values)
        indices = [i for i, b in enumerate(basis) if b['level_id'] == level.level_id]
        coupled = [(F, mf) for F in range(abs(level.two_J-I2), level.two_J+I2+1, 2)
                   for mf in range(-F, F+1, 2)]
        V = np.array([[_cg(level.two_J, basis[i]['two_mJ'], I2, basis[i]['two_mI'], F, mf)
                       for F, mf in coupled] for i in indices])
        J, I = level.two_J/2, I2/2
        shifts = []
        for F2, _ in coupled:
            F = F2/2
            K = F*(F+1)-I*(I+1)-J*(J+1)
            shift = Ahz*K/2
            if I >= 1 and J >= 1:
                shift += Bhz*(.75*K*(K+1)-I*(I+1)*J*(J+1))/(2*I*(2*I-1)*J*(2*J-1))
            shifts.append(shift/(100*model.constants.c))
        H0[np.ix_(indices, indices)] += (V*np.asarray(shifts)) @ V.T
    nuclear_g_bohr = (mu/(I2/2)*model.constants.m_e/model.constants.m_p) if I2 else 0.
    T0 = np.kron(model.T0, np.eye(I2+1))-np.diag([nuclear_g_bohr*b['two_mI']/2 for b in basis])
    flags = tuple(x for x in model.validity_flags if x != 'HYPERFINE_UNASSESSED')+('HFS_OFFDIAGONAL_NOT_INCLUDED',)
    metadata = dict(model.metadata, hyperfine=asdict(hfs), projection_definition='M_F')
    return AtomicModel(model.model_id, model.levels, basis, H0, T0,
                       {q: np.kron(d, np.eye(I2+1)) for q, d in model.Dq.items()},
                       np.kron(model.known_pairs, np.ones((I2+1, I2+1), bool)).astype(bool),
                       model.constants, flags, metadata).validate()


class AtomicComponents(dict):
    """Physical component columns, a state table, diagnostics and provenance."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def directional_rates(self, theta_deg, *, phi_deg=0., analyzer=None, transmissions=(1., 1.)):
        """dA/dOmega in s^-1 sr^-1, with an optional normalized complex Jones pair.

Jones coordinates refer to the transverse e_theta/e_phi basis. Unpolarized
detection sums both axes. Unlike total A, this output depends on geometry.
"""
        theta = np.deg2rad(_finite(theta_deg, 'theta_deg'))
        phi = np.deg2rad(_finite(phi_deg, 'phi_deg'))
        transmission = np.asarray(transmissions, float)
        if transmission.shape != (2,) or np.any(~np.isfinite(transmission)) or np.any(transmission < 0):
            raise ValueError('transmissions must be two finite nonnegative values')
        d = self.complex_dipole_components
        # Cartesian <l|D|u> is the conjugate of the Cartesian <u|D|l>.
        cartesian = np.column_stack(((d[:, 0]-d[:, 2])/np.sqrt(2),
                                     1j*(d[:, 0]+d[:, 2])/np.sqrt(2), d[:, 1])).conj()
        e_theta = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
        e_phi = np.array([-np.sin(phi), np.cos(phi), 0.])
        projected = np.column_stack((cartesian @ e_theta, cartesian @ e_phi))*np.sqrt(transmission)
        if analyzer is None:
            strength = np.sum(np.abs(projected)**2, axis=1)
        else:
            jones = np.asarray(analyzer, complex)
            if jones.shape != (2,) or not np.all(np.isfinite(jones)) or not np.isclose(np.vdot(jones, jones), 1):
                raise ValueError('analyzer must be a normalized transverse complex Jones pair')
            strength = np.abs(projected @ jones.conj())**2
        return e1_rate(self.nu_Hz, strength, constants=self.constants)*3/(8*np.pi)

    def emission_weights(self, populations, *, power=False, theta_deg=None, **geometry):
        """Multiply rates by explicit incoherent state populations and optional h*nu.

populations is an array aligned with states['state_id'], or a mapping
covering every emitting upper state ID. No default population is assumed.
"""
        if isinstance(populations, dict):
            missing = set(self.upper_state_id)-set(populations)
            if missing:
                raise MissingAtomicData(f'Missing upper-state populations: {sorted(missing)}')
            values = np.array([populations[x] for x in self.upper_state_id], float)
        else:
            values = np.asarray(populations, float)
            if values.shape != self.states['energy_cm'].shape:
                raise ValueError('Population array must match the state table')
            values = values[self.upper_index]
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError('Populations must be finite and nonnegative')
        rates = self.A_s1 if theta_deg is None else self.directional_rates(theta_deg, **geometry)
        if theta_deg is None and geometry:
            raise ValueError('Geometry options require theta_deg')
        return values*rates*(self.constants.h*self.nu_Hz if power else 1.)


def calculate_components(model, field_T):
    """Solve parity/M blocks and export every allowed positive-frequency E1 pair.

Norm checks precede emission filtering. Zero-strength allowed components are
retained. Partial decay sums are not lifetimes; no population or geometry
normalization is applied to atomic strengths or rates.
"""
    model.validate()
    B = _finite(field_T, 'field_T')
    H = model.H0 + model.constants.mu_B_cm*B*model.T0
    groups = {}
    for i, b in enumerate(model.basis):
        groups.setdefault((b['parity'], b['two_Mtotal']), []).append(i)
    energies, vectors, projections, parities = [], [], [], []
    for (parity, m), indices in groups.items():
        block = H[np.ix_(indices, indices)]
        origin = float(np.trace(block).real/len(indices))
        shifted, C = np.linalg.eigh(block-origin*np.eye(len(indices)))
        full = np.zeros((len(model.basis), len(indices)), complex)
        full[indices] = C
        energies.extend(shifted+origin)
        vectors.append(full)
        projections.extend([m]*len(indices))
        parities.extend([parity]*len(indices))
    order = np.argsort(energies, kind='stable')
    E = np.asarray(energies)[order]
    C = np.concatenate(vectors, axis=1)[:, order]
    m, parity = np.asarray(projections)[order], np.asarray(parities)[order]
    D = np.stack([C.conj().T @ model.Dq[q] @ C for q in (-1, 0, 1)], axis=-1)
    strengths = np.sum(np.abs(D)**2, axis=-1)
    original_norm = sum(np.linalg.norm(x, 'fro')**2 for x in model.Dq.values())
    dressed_norm = float(np.sum(strengths))
    weights = np.abs(C)**2
    requested = weights.T @ model.known_pairs @ weights > 1e-12
    difference = E[:, None]-E
    mask = (difference > 0) & (parity[:, None] != parity) & (np.abs(m[:, None]-m) <= 2) & requested
    ui, li = np.where(mask)
    nu = difference[ui, li]*100*model.constants.c
    strength = strengths[ui, li]
    rates = e1_rate(nu, strength, constants=model.constants)
    ids = np.array([f'state_{i:04d}' for i in range(len(E))])
    level_ids = tuple(x.level_id for x in model.levels)
    overlaps = np.column_stack([np.sum(weights[[i for i, b in enumerate(model.basis) if b['level_id'] == level_id]], axis=0)
                                for level_id in level_ids])
    residual = float(np.max(np.abs(H @ C-C*E)))
    flags = list(model.validity_flags)
    if np.any(np.diff(E) <= max(1., np.max(np.abs(E)))*1e-12):
        flags.append('DEGENERATE_STATE_IDENTITIES')
    state_table = dict(state_id=ids, energy_cm=E, energy_unit='cm^-1', two_Mtotal=m, parity=parity,
                       eigenvectors=C, basis=model.basis, parent_level_ids=level_ids,
                       parent_level_overlaps=overlaps, partial_decay_rate_s1=np.bincount(ui, weights=rates, minlength=len(E)))
    diagnostics = dict(max_eigen_residual_cm=residual, max_unitarity_error=float(np.max(np.abs(C.conj().T @ C-np.eye(len(E))))),
                       input_strength_sum_au=float(original_norm), full_strength_sum_au=dressed_norm,
                       relative_norm_error=abs(dressed_norm-original_norm)/original_norm if original_norm else 0.)
    return AtomicComponents(field_T=B, upper_state_id=ids[ui], lower_state_id=ids[li], upper_index=ui, lower_index=li,
                             upper_energy=E[ui], lower_energy=E[li], energy_unit='cm^-1',
                             upper_two_Mtotal=m[ui], lower_two_Mtotal=m[li], upper_parity=parity[ui], lower_parity=parity[li],
                             total_projection_definition=model.metadata.get('projection_definition', 'M_J'),
                             q=(m[ui]-m[li])//2, nu_Hz=nu, lambda_vac_nm=model.constants.c/nu*1e9,
                             strength_au=strength, strength_SI=strength*model.constants.strength_unit_SI,
                             complex_dipole_components=D[ui, li], dipole_component_q=(-1, 0, 1),
                             A_s1=rates, multipole='E1', model_id=model.model_id,
                             basis_id=model.metadata.get('matrix_dataset_id', 'complete retained angular basis'),
                             source_ids=model.metadata.get('source_ids', []), uncertainty_or_sensitivity_bounds=None,
                             validity_flags=tuple(dict.fromkeys(flags)), states=state_table, diagnostics=diagnostics,
                             metadata=dict(model.metadata), constants=model.constants)


def rephase_model(model, phases):
    """Apply a consistent arbitrary phase to each basis state and every operator."""
    p = np.asarray(phases, complex)
    if p.shape != (len(model.basis),) or not np.all(np.isfinite(p)) or not np.allclose(np.abs(p), 1):
        raise ValueError('Provide one unit-modulus phase per basis state')
    def transform(matrix):
        return p.conj()[:, None]*matrix*p[None, :]
    return replace(model, H0=transform(model.H0), T0=transform(model.T0),
                   Dq={q: transform(d) for q, d in model.Dq.items()}).validate()


def calculate_field_grid(model, fields_T, *, track=True):
    """Calculate a supplied grid and optionally track IDs by symmetry-block overlap.

Degenerate state identities remain flagged. Tracking manages stationary
labels; it does not prescribe dynamical populations or assess convergence.
"""
    from scipy.optimize import linear_sum_assignment
    fields = np.asarray(fields_T, float)
    if fields.ndim != 1 or not len(fields) or not np.all(np.isfinite(fields)):
        raise ValueError('fields_T must be a finite nonempty one-dimensional grid')
    results = []
    for B in fields:
        current = calculate_components(model, B)
        if track and results:
            previous = results[-1]
            p, c = previous.states, current.states
            similarity = np.abs(p['eigenvectors'].conj().T @ c['eigenvectors'])**2
            compatible = (p['two_Mtotal'][:, None] == c['two_Mtotal']) & (p['parity'][:, None] == c['parity'])
            rows, cols = linear_sum_assignment(np.where(compatible, -similarity, 1e6))
            ids = c['state_id'].copy()
            ids[cols] = p['state_id'][rows]
            c['state_id'] = ids
            current['upper_state_id'], current['lower_state_id'] = ids[current.upper_index], ids[current.lower_index]
        results.append(current)
    return results


def components_from_inputdeck(deck, radiative_data):
    """Opt-in Model B bridge for a legacy input dictionary.

RadiativeData contains references with upper_two_J/lower_two_J and the
RadiativeReference datum fields, or orbital_strength_au. AtomicComponents
uses the new consistent constants; old relative outputs remain unchanged.
"""
    if not isinstance(radiative_data, dict):
        raise ValueError('RadiativeData must be a dictionary')
    unknown = set(radiative_data)-{'references', 'orbital_strength_au', 'consistency_rtol', 'isotope_id'}
    if unknown:
        raise ValueError(f'Unknown RadiativeData options: {sorted(unknown)}')
    def term(which, parity):
        S, L = deck[f's_{which}'], deck[f'l_{which}']
        if 2*S != round(2*S) or L != round(L):
            raise ValueError('RadiativeData requires exact half-integer spin and integer orbital momentum')
        jvalues = range(abs(2*int(L)-int(2*S)), 2*int(L)+int(2*S)+1, 2)
        values = np.atleast_1d(np.asarray(deck[f'E_{which}'], float))
        if values.shape != (len(jvalues),) or not np.all(np.isfinite(values)):
            raise MissingAtomicData(f'RadiativeData needs all energies for the {which} term')
        energies = dict(zip(jvalues, values))
        return LSTerm(which, int(L), int(2*S), energies, parity)
    lower, upper = term('ground', 1), term('excited', -1)
    records = []
    for original in radiative_data.get('references', []):
        record = dict(original)
        uj, lj = record.pop('upper_two_J'), record.pop('lower_two_J')
        _two(uj, 'upper_two_J'); _two(lj, 'lower_two_J')
        records.append(RadiativeReference(f'excited:J{uj}', f'ground:J{lj}', **record))
    model = build_model_b(lower, upper, records, orbital_strength_au=radiative_data.get('orbital_strength_au'),
                          consistency_rtol=radiative_data.get('consistency_rtol', 1e-6))
    model.metadata['legacy_input_bridge'] = 'Separate atomic calculation; legacy relative spectra and constants retained'
    if 'I_spin' in deck:
        I = deck['I_spin']
        if 2*I != round(2*I):
            raise ValueError('I_spin must be an exact half integer')
        constants = {}
        for side, label in ((lower, 'G'), (upper, 'E')):
            A = np.broadcast_to(np.asarray(deck[f'HFS_{label}'], float), (len(side.energies_cm),))
            Q = np.broadcast_to(np.asarray(deck.get(f'HFSB_{label}', 0.), float), A.shape)
            constants.update({f'{side.term_id}:J{j}': (a, b) for j, a, b in zip(sorted(side.energies_cm), A, Q)})
        isotope = radiative_data.get('isotope_id')
        if not isotope:
            raise MissingAtomicData('RadiativeData with HFS requires isotope_id')
        model = add_hyperfine(model, HyperfineData(int(2*I), deck['mu_I'], constants, isotope))
    if deck.get('Efield', 0.) != 0:
        from .stark import _polarizabilities
        polarizabilities = {}
        for side, suffix in ((lower, 'G'), (upper, 'E')):
            js = [j/2 for j in sorted(side.energies_cm)]
            a0 = _polarizabilities(deck.get(f'alpha0_{suffix}'), js, f'alpha0_{suffix}')
            a2 = _polarizabilities(deck.get(f'alpha2_{suffix}'), js, f'alpha2_{suffix}', default=0.)
            polarizabilities.update({f'{side.term_id}:J{int(2*j)}': (a0[j], a2[j]) for j in js})
        model = add_dc_stark(model, deck['Efield'], polarizabilities)
    return calculate_components(model, deck['Bmag'])


def synthesize_spectrum(components, wavelength_vac_nm, populations, *, profile='GaussianInstrum',
                        temperature_K=300., mass_amu=1., width_nm=.002, power=False, theta_deg=None, **geometry):
    """Sum population-weighted wavelength profiles without strength pruning.

Return a density per nm on an explicit vacuum-wavelength grid. A finite
grid loses tails; captured_weight reports its numerical integral. The
profile layer uses the existing Doppler/instrument implementations and does
not add Stark dynamics, redistribution, or radiative transfer.
"""
    from .lineshapes import Gaussian, GaussianInstrum, Lorrentzian, Voigt
    grid = np.asarray(wavelength_vac_nm, float)
    if grid.ndim != 1 or len(grid) < 2 or np.any(~np.isfinite(grid)) or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise ValueError('Supply an increasing positive finite wavelength grid with at least two points')
    _finite(mass_amu, 'mass_amu', positive=True)
    _finite(width_nm, 'width_nm', positive=True)
    _finite(temperature_K, 'temperature_K', minimum=0)
    if profile == 'Gaussian' and temperature_K == 0:
        raise ValueError('A Doppler Gaussian requires positive temperature')
    functions = {'Gaussian': Gaussian, 'GaussianInstrum': GaussianInstrum, 'Voigt': Voigt, 'Lorrentzian': Lorrentzian}
    if profile not in functions:
        raise ValueError('Unknown atomic spectrum profile')
    weights = components.emission_weights(populations, power=power, theta_deg=theta_deg, **geometry)
    density = np.zeros_like(grid)
    for center, weight in zip(components.lambda_vac_nm, weights):
        if weight == 0:
            continue
        x, x0 = grid*1e-9, center*1e-9
        if profile == 'Gaussian':
            values = Gaussian(x, x0, temperature_K, mass_amu)
        elif profile == 'Lorrentzian':
            values = Lorrentzian(x, x0, pixel_size=width_nm*1e-9)
        else:
            values = functions[profile](x, x0, temperature_K, mass_amu, pixel_size=width_nm*1e-9)
        density += weight*values*1e-9
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return dict(wavelength_vac_nm=grid, density_per_nm=density, expected_weight=float(np.sum(weights)),
                captured_weight=float(integrate(density, grid)), power=power,
                directional=theta_deg is not None, profile=profile,
                validity_flags=components.validity_flags+('FINITE_PROFILE_WINDOW', 'BROADENING_MODEL_UNASSESSED'))


def frequency_profile_to_wavelength(wavelength_m, profile_nu, *, constants=None):
    """Convert a density per Hz to a density per metre using c/lambda^2."""
    k = constants or AtomicConstants()
    grid = np.asarray(wavelength_m, float)
    if np.any(~np.isfinite(grid)) or np.any(grid <= 0):
        raise ValueError('Wavelengths must be finite and positive')
    values = np.asarray(profile_nu(k.c/grid), float)
    if values.shape != grid.shape or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError('The frequency profile must return matching finite nonnegative densities')
    return k.c/grid**2*values


def isolated_multipole_rates(upper, lower, field_T, A_reference_s1, frequency_reference_Hz, *, multipole='E1', constants=None):
    """Isolated-J components of an identified E1/M1/E2/M2 reference rate.

No E1 absolute-strength units are assigned to other multipoles. The
long-wavelength frequency exponent is 2k+1. Combined M1+E2 records are
insufficient and rejected; general mixed-multipole matrices are not inferred.
"""
    k = constants or AtomicConstants()
    u = upper if isinstance(upper, AtomicLevel) else AtomicLevel(**upper)
    l = lower if isinstance(lower, AtomicLevel) else AtomicLevel(**lower)
    _levels([u, l])
    if multipole not in ('E1', 'M1', 'E2', 'M2'):
        raise ValueError('Supply one identified E1, M1, E2 or M2 contribution')
    rank = int(multipole[1])
    parity_product = (-1)**(rank if multipole[0] == 'E' else rank+1)
    if u.parity*l.parity != parity_product or not _rank_allowed(u.two_J, l.two_J, rank):
        raise ValueError('Multipole parity or angular selection rule is violated')
    if u.g_J is None or l.g_J is None:
        raise MissingAtomicData('Isolated-J multipoles require both g_J values')
    B = _finite(field_T, 'field_T')
    rate = _finite(A_reference_s1, 'A_reference_s1', minimum=0)
    source_nu = _finite(frequency_reference_Hz, 'frequency_reference_Hz', positive=True)
    rows = []
    for mu in range(-u.two_J, u.two_J+1, 2):
        for ml in range(-l.two_J, l.two_J+1, 2):
            q2 = mu-ml
            if abs(q2) > 2*rank:
                continue
            difference = u.energy_cm-l.energy_cm + k.mu_B_cm*B*(u.g_J*mu-l.g_J*ml)/2
            if difference <= 0:
                continue
            angular = _w3(u.two_J, 2*rank, l.two_J, -mu, q2, ml)**2
            nu = difference*100*k.c
            rows.append((mu, ml, q2//2, nu, angular, rate*(nu/source_nu)**(2*rank+1)*(u.two_J+1)*angular))
    columns = np.asarray(rows, float).reshape(-1, 6)
    return dict(upper_two_M=columns[:, 0].astype(int), lower_two_M=columns[:, 1].astype(int),
                q=columns[:, 2].astype(int), nu_Hz=columns[:, 3], angular_strength=columns[:, 4],
                A_s1=columns[:, 5], multipole=multipole, field_T=B,
                strength_unit='dimensionless angular factor; no multipole-specific absolute strength inferred',
                validity_flags=_flags(('ISOLATION_UNASSESSED',)))
