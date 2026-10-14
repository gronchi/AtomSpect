# -*- coding: utf-8 -*-
"""Public AtomSpect API for relative Zeeman and hyperfine E1 spectra.

Import AtomSpect_Main, SpectrumResult, and the legacy calculation/plotting
helpers from this module. Implementations live in the corresponding package
modules; see doc/README.md for the manual, examples, and validity limits.
"""

import numpy as np
import scipy as scp
import csv
from matplotlib import pyplot as plt
from matplotlib import ticker
import tkinter as tk
from tkinter import simpledialog

from matplotlib.widgets import Button, Slider

from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_3j as w3j, wigner_6j as w6j

from .constants import *  # noqa: F401,F403
from .units import *  # noqa: F401,F403
from .lineshapes import *  # noqa: F401,F403
from .convolution import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .plotting import *  # noqa: F401,F403
from .angular import *  # noqa: F401,F403
from .hfs import *  # noqa: F401,F403
from .radiative import *  # noqa: F401,F403
from .polarization import *  # noqa: F401,F403
from .hamiltonian import *  # noqa: F401,F403
from .levels import *  # noqa: F401,F403
from .transitions import *  # noqa: F401,F403
from .fan import *  # noqa: F401,F403
from .polarization import _eigenpair_polarization  # noqa: F401
from .hamiltonian import _magnetic_eigenvectors, _hfs_magnetic_eigenvectors  # noqa: F401
from .transitions import _zeeman_signal  # noqa: F401
from .config import InputDeck, _approximation_enabled  # noqa: F401
from .spectrum import AtomSpect_Main  # noqa: F401
from .result import SpectrumResult  # noqa: F401
