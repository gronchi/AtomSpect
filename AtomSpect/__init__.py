# -*- coding: utf-8 -*-
"""
AtomSpect main function definitions.
@author: Leo Nofs
This contains all of the functions for operating the code. This file is NOT run directly.
For details, updates, questions, bug reports, refer to the project github: https://github.com/Lnofs/AtomSpect


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
