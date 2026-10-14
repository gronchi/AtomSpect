"""External atomic data and E1 conversions for the revised Zeeman guide.

New calculations use a consistent SciPy constants set. Historical constants
and radiative placeholders remain unchanged for compatibility.
"""
from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Integral

import numpy as np
import scipy
from scipy import constants as sc
from sympy import Rational
from sympy.physics.wigner import wigner_3j, wigner_6j, clebsch_gordan

__all__ = ['AtomicConstants', 'AtomicLevel', 'RadiativeReference', 'LSTerm',
           'HyperfineData', 'MissingAtomicData', 'e1_rate', 'e1_line_strength',
           'einstein_b', 'fit_ls_norm']


class MissingAtomicData(ValueError):
    """Required operator information is unknown rather than symmetry-zero."""


def _finite(value, name, minimum=None, positive=False):
    if not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite scalar')
    if minimum is not None and value < minimum or positive and value <= 0:
        raise ValueError(f'Invalid {name}: {value}')
    return float(value)


def _two(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer (twice the angular momentum)')
    return int(value)


@lru_cache(maxsize=None)
def _w3(a, b, c, ma, mb, mc):
    return float(wigner_3j(*[Rational(x, 2) for x in (a, b, c, ma, mb, mc)]))


@lru_cache(maxsize=None)
def _w6(a, b, c, d, e, f):
    return float(wigner_6j(*[Rational(x, 2) for x in (a, b, c, d, e, f)]))


@lru_cache(maxsize=None)
def _cg(a, ma, b, mb, c, mc):
    return float(clebsch_gordan(*[Rational(x, 2) for x in (a, b, c, ma, mb, mc)]))


def _rank_allowed(a, b, rank=1):
    return a % 2 == b % 2 and abs(a-b) <= 2*rank <= a+b


@dataclass(frozen=True)
class AtomicConstants:
    """SI constants and derived atomic-unit factors, with a recorded version.

mu_B_cm and C_E1 may be overridden to reproduce a frozen benchmark; the
override then defines the magnetic scale and rate conversion for that run.
"""
    h: float = sc.h
    c: float = sc.c
    epsilon_0: float = sc.epsilon_0
    e: float = sc.e
    m_e: float = sc.m_e
    m_p: float = sc.m_p
    a_0: float = sc.physical_constants['Bohr radius'][0]
    g_s: float = abs(sc.physical_constants['electron g factor'][0])
    mu_B_cm: float = sc.e/(4*np.pi*sc.m_e*sc.c*100)
    C_E1: float | None = None
    source: str = f'scipy.constants {scipy.__version__}'

    def __post_init__(self):
        for name in ('h', 'c', 'epsilon_0', 'e', 'm_e', 'm_p', 'a_0', 'g_s', 'mu_B_cm'):
            _finite(getattr(self, name), name, positive=True)
        if self.C_E1 is None:
            value = 16*np.pi**3*self.strength_unit_SI/(3*self.epsilon_0*self.h)*1e30
            object.__setattr__(self, 'C_E1', value)
        _finite(self.C_E1, 'C_E1', positive=True)

    @property
    def strength_unit_SI(self):
        """One atomic-unit E1 strength, (e*a0)^2, in C^2 m^2."""
        return (self.e*self.a_0)**2


@dataclass(frozen=True)
class AtomicLevel:
    """One zero-field level; energies are cm^-1 and parity is +1 or -1."""
    level_id: str
    energy_cm: float
    two_J: int
    parity: int
    g_J: float | None = None
    source_id: str = 'user-supplied'
    energy_reference_id: str = 'common-zero'
    species_id: str = 'unspecified'
    isotope_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.level_id or not self.source_id or not self.energy_reference_id:
            raise ValueError('Level IDs, sources, and energy-reference IDs must be nonempty')
        _finite(self.energy_cm, 'energy_cm')
        _two(self.two_J, 'two_J')
        if isinstance(self.parity, bool) or self.parity not in (-1, 1):
            raise ValueError('parity must be +1 or -1')
        if self.g_J is not None:
            _finite(self.g_J, 'g_J')


@dataclass(frozen=True)
class RadiativeReference:
    """A zero-field E1 datum with its original frequency and provenance.

kind is A (s^-1), f, gf, loggf, strength_au, strength_SI, or lifetime
(seconds, requiring branching_fraction). Reduced strengths need no frequency;
all other kinds require a positive source vacuum frequency or wavelength.
"""
    upper_level_id: str
    lower_level_id: str
    kind: str
    value: float
    wavelength_vac_nm: float | None = None
    frequency_Hz: float | None = None
    branching_fraction: float | None = None
    relative_uncertainty: float = 0.
    source_id: str = 'user-supplied'
    multipole: str = 'E1'
    metadata: dict = field(default_factory=dict)
    upper_two_F: int | None = None
    lower_two_F: int | None = None
    two_I: int | None = None

    def frequency(self, constants):
        """Source frequency; never substitute field-dressed or air wavelengths."""
        if self.frequency_Hz is not None:
            nu = _finite(self.frequency_Hz, 'frequency_Hz', positive=True)
            if self.wavelength_vac_nm is not None:
                other = constants.c/(_finite(self.wavelength_vac_nm, 'wavelength_vac_nm', positive=True)*1e-9)
                if not np.isclose(nu, other, rtol=1e-10):
                    raise ValueError('Source vacuum wavelength and frequency disagree')
            return nu
        if self.wavelength_vac_nm is not None:
            return constants.c/(_finite(self.wavelength_vac_nm, 'wavelength_vac_nm', positive=True)*1e-9)
        raise MissingAtomicData('This radiative datum requires its source vacuum frequency or wavelength')

    def strength_au(self, upper, lower, constants=None):
        """Convert the original datum into a level-reduced E1 strength."""
        constants = constants or AtomicConstants()
        if self.multipole != 'E1':
            raise ValueError('E1 inversion cannot be applied to another or blended multipole')
        if self.upper_level_id != upper.level_id or self.lower_level_id != lower.level_id:
            raise ValueError('Radiative reference level IDs do not match')
        _finite(self.relative_uncertainty, 'relative_uncertainty', minimum=0)
        if not self.source_id:
            raise ValueError('Radiative source_id must be nonempty')
        upper_weight_j, lower_weight_j, recoupling = upper.two_J, lower.two_J, 1.
        hfs_values = (self.upper_two_F, self.lower_two_F, self.two_I)
        if any(x is not None for x in hfs_values):
            if any(x is None for x in hfs_values):
                raise MissingAtomicData('A hyperfine reference needs upper_two_F, lower_two_F and two_I')
            Fu, Fl, I = (_two(x, name) for x, name in zip(hfs_values, ('upper_two_F', 'lower_two_F', 'two_I')))
            if Fu not in range(abs(upper.two_J-I), upper.two_J+I+1, 2) or Fl not in range(abs(lower.two_J-I), lower.two_J+I+1, 2):
                raise ValueError('Hyperfine reference F is incompatible with electronic J and nuclear I')
            if not _rank_allowed(Fu, Fl):
                raise ValueError('Hyperfine reference violates E1 angular selection rules')
            recoupling = (Fu+1)*(Fl+1)*_w6(upper.two_J, Fu, I, Fl, lower.two_J, 2)**2
            if recoupling <= 0:
                raise ValueError('A forbidden hyperfine branch cannot normalize an electronic line')
            upper_weight_j, lower_weight_j = Fu, Fl
        if self.kind == 'loggf':
            _finite(self.value, 'loggf')
            with np.errstate(over='ignore'):
                value = np.power(10., self.value)
            _finite(value, 'gf', minimum=0)
        else:
            value = _finite(self.value, self.kind, minimum=0)
        if self.kind == 'strength_au':
            return value/recoupling
        if self.kind == 'strength_SI':
            return value/constants.strength_unit_SI/recoupling
        nu = self.frequency(constants)
        if self.kind == 'lifetime':
            _finite(value, 'lifetime', positive=True)
            fraction = _finite(self.branching_fraction, 'branching_fraction', minimum=0)
            if fraction > 1:
                raise ValueError('branching_fraction cannot exceed one')
            value = fraction/value
        elif self.kind in ('f', 'gf', 'loggf'):
            gf = value*(lower_weight_j+1) if self.kind == 'f' else value
            value = 2*np.pi*constants.e**2*nu**2/(constants.m_e*constants.c**3*constants.epsilon_0)*gf/(upper_weight_j+1)
        elif self.kind != 'A':
            raise ValueError(f'Unknown radiative datum kind: {self.kind}')
        return e1_line_strength(nu, value, upper_weight_j, constants=constants)/recoupling


@dataclass(frozen=True)
class LSTerm:
    """A complete LS term; energies_cm maps exact two_J integers to energies."""
    term_id: str
    L: int
    two_S: int
    energies_cm: dict
    parity: int
    source_id: str = 'user-supplied'
    species_id: str = 'unspecified'
    energy_reference_id: str = 'common-zero'
    metadata: dict = field(default_factory=dict)

    def levels(self):
        _two(self.L, 'L')
        _two(self.two_S, 'two_S')
        expected = set(range(abs(2*self.L-self.two_S), 2*self.L+self.two_S+1, 2))
        for value in self.energies_cm:
            _two(value, 'energy key two_J')
        if set(self.energies_cm) != expected:
            raise MissingAtomicData(f'{self.term_id} requires every J energy: two_J={sorted(expected)}')
        if not self.term_id:
            raise ValueError('term_id must be nonempty')
        return [AtomicLevel(f'{self.term_id}:J{j}', self.energies_cm[j], j, self.parity,
                            source_id=self.source_id, species_id=self.species_id,
                            energy_reference_id=self.energy_reference_id,
                            metadata=dict(self.metadata, term_id=self.term_id, L=self.L, two_S=self.two_S))
                for j in sorted(expected)]


@dataclass(frozen=True)
class HyperfineData:
    """One isotope; constants_Hz maps every level ID to (A_Hz, B_Hz).

Only diagonal electronic-J hyperfine constants are represented. Missing
constants are unknown; supply explicit zeros for known zero contributions.
"""
    two_I: int
    mu_I: float
    constants_Hz: dict
    isotope_id: str
    source_id: str = 'user-supplied'


def e1_rate(frequency_Hz, strength, *, units='au', constants=None):
    """Angle-integrated rate per specified magnetic state; no weight division."""
    k = constants or AtomicConstants()
    nu, s = np.asarray(frequency_Hz, float), np.asarray(strength, float)
    if np.any(~np.isfinite(nu)) or np.any(nu <= 0) or np.any(~np.isfinite(s)) or np.any(s < 0):
        raise ValueError('Positive finite frequencies and nonnegative finite strengths are required')
    if units == 'SI':
        s = s/k.strength_unit_SI
    elif units != 'au':
        raise ValueError('strength units must be au or SI (C^2 m^2)')
    return k.C_E1*(nu/k.c*1e-10)**3*s


def e1_line_strength(frequency_Hz, A_s1, upper_two_J, *, constants=None):
    """Infer a zero-field level-reduced strength in au, including upper weight."""
    _two(upper_two_J, 'upper_two_J')
    rate = np.asarray(A_s1, float)
    if np.any(~np.isfinite(rate)) or np.any(rate < 0):
        raise ValueError('A_s1 must be finite and nonnegative')
    return rate*(upper_two_J+1)/e1_rate(frequency_Hz, 1., constants=constants)


def einstein_b(frequency_Hz, A_s1, *, constants=None, lower_weight=None, upper_weight=None):
    """Stimulated coefficient for energy density per ordinary frequency.

Without weights this is the specified-state B_ul=B_lu. Supplying both
weights returns (B_ul, B_lu) for level-averaged coefficients.
"""
    k = constants or AtomicConstants()
    nu, rate = np.asarray(frequency_Hz, float), np.asarray(A_s1, float)
    if np.any(~np.isfinite(nu)) or np.any(nu <= 0) or np.any(~np.isfinite(rate)) or np.any(rate < 0):
        raise ValueError('Invalid frequency or rate')
    value = k.c**3/(8*np.pi*k.h*nu**3)*rate
    if lower_weight is None and upper_weight is None:
        return value
    _finite(lower_weight, 'lower_weight', positive=True)
    _finite(upper_weight, 'upper_weight', positive=True)
    return value, value*upper_weight/lower_weight


def fit_ls_norm(angular_factors, strengths_au, covariance):
    """Fit R^2 and its variance using an explicitly supplied covariance matrix."""
    factors, strengths, cov = map(lambda x: np.asarray(x, float), (angular_factors, strengths_au, covariance))
    if factors.ndim != 1 or strengths.shape != factors.shape or cov.shape != (len(factors), len(factors)):
        raise ValueError('Incompatible LS fit arrays')
    if not len(factors) or any(not np.all(np.isfinite(x)) for x in (factors, strengths, cov)):
        raise ValueError('LS fit data must be nonempty and finite')
    if np.any(factors <= 0) or np.any(strengths < 0) or not np.allclose(cov, cov.T):
        raise ValueError('Invalid LS factors, strengths, or covariance')
    np.linalg.cholesky(cov)
    inverse_factors = np.linalg.solve(cov, factors)
    denominator = factors @ inverse_factors
    return float(inverse_factors @ strengths/denominator), float(1/denominator)
