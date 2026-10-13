"""Hyperfine interval formula."""
import numpy as np

from .constants import c_light


def Delta_E_HFS(I, J, F, A, B):
    #Calculate the shift in energy due to the hyperfine effect, where A and B are the hyperfine constants.
    #These are in units of Hz typically. Output is in units of cm^-1
    #1/(100*c_light) is the conversion factor that takes the Hz and results in cm^-1
    #See SteckRb for a good overview of hyperfine. Kopfermann is a very detailed source as well.
    
    # First need to define a constant K that will be used in the calculations
    K = F*(F+1) - I*(I+1) - J*(J+1)
    #Calculate A_hfs
    E_Ahfs = 0.5*A*K / (100*c_light)
    #Check for the cases when the B_hfs contribution must be zero (I=0,1/2, J=0,1/2, or if B=0 )
    if I == np.abs(0.5) or J == 0.5 or I == 0 or J == 0 or B == 0:
        E_Bhfs = 0
    else:
        #If none of those cases are true, calculate the shift from the quadropole term.
        E_Bhfs = B*((0.75*K*(K+1) - I*(I+1)*J*(J+1))/(2*I*(2*I - 1)*J*(2*J - 1)))/(100*c_light)
    return E_Ahfs + E_Bhfs
