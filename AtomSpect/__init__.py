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


# %%Main


def _approximation_enabled(value):
    """Read Boolean flags and legacy yes/no strings consistently."""
    if value is None:
        return False
    if isinstance(value, str):
        answer = value.strip().lower()
        if answer in ('y', 'yes', 'true', '1'):
            return True
        if answer in ('n', 'no', 'false', '0', ''):
            return False
        raise ValueError('Approximation flags require Boolean or yes/no values')
    if np.isscalar(value) and value in (0, 1):
        return bool(value)
    raise ValueError('Approximation flags require Boolean or yes/no values')


def AtomSpect_Main(Inputdeck):
    '''
    This is the main zeeman function. THe input is a dictionary with many optional keywords for more modular functionaly. These optional keywords are checked for to branch the function.


    Parameters
    ----------
    Inputdeck :     InputdeckExample = {
                     's_ground': 3  , #Spin multiplicity for ground state ^(2s +1)L_J , int or half int
                     's_excited': 3 , #Spin multiplicity for excited state ^(2s +1)L_J, int or half int
                     'l_ground': 0  , #Orbital Angular Momentum of ground state, int or half int
                     'l_excited': 1 , #Orbital Angular Momentum of excited state, int or half int
                     'E_ground': np.array([2951.29 ]) , #np.array of lower state, Lowest J first, cm^-1
                     'E_excited': np.array([26229.77,27488.11,27889.68]) ,  #np.array of upper state, Lowest J first, cm^-1
                     'Bmag': .6 , #Magnetic Field  [T]
                     'b_angle': 60 , #Angle between LoS and Bmax
                     'Pol_angle': 0 , #Angle polarizing filter makes with max linear transmission
                     'amu' : 183 , #Weight in AMU
                     'Convfxn': 'Gaussian' , #Optional: 'Gaussian', 'GaussianInstrum'', 'Skewed' , 'Lorrentzian', 'Voigt'.
                     'Skewness' : [0.62,0.99] #Optional, [Left,Right] constant for the skewness of the lorrentzian function. Between 0 and 1. If both are 1, it is a normal lorrentzian.
                     'Temp' : 300, #Temperature in K. Used for Gaussian, Gaussian Instrum, Voigt convolutions
                     'TempeV' : 20 , #Temperature in eV. Can be passed instead of Temp in K. Converted as needed internally.
                     'specstep' : 0.002222 , #stepsize of linefunction [nm]
                     'specres' : 10 , #How many points are evaluated per spectrometer pixel window. Higher means smoother curve.
                     'fxnwindow': 0.5 , #How far from the central peak the convolution will be calculated. Also related to how stick binning works. 
                     'plottitle': 'Excample Title' , #Title for plotting (optional).                      
                     'SpectrumData' : [[Wavelength],[Intensity]] , #List of arrays for wavelength (nm) and intensity (arb)
                     'spec_window' : [425,435], #Min and max for convolution window (nm)
                     'plot_window' : [425,435], #Min and max for plot window range (nm)
                     'HFS_G' : [505.5e6] , #Optional: Hyperfine A constant for the lower level, need one entry per J level, lowest J first [Hz]
                     'HFS_E': [0, 496.2e6, 440.5e6] , #Optional: Hyperfine A constants for upper level, need one entery per J level, lowest J first [Hz]
                     'I_spin' : 0.5, #Optional: nuclear spin, I. Tabulated by Stone et al.
                     'mu_I' :  .11778476 , #Optional: Nuclear dipole moment as tabulated by Stone.              
                     'HFSB_G' : [505.5e6] , #OPtional: Hyperfine B constant for the lower level, need one entry per J level, lowest J first [Hz]
                     'HFSB_E': [0, 496.2e6, 440.5e6] , #Optional: Hyperfine B constants for upper level, need one entery per J level, lowest J first [Hz]
                     'Calc_HFS_Vals': [False,False], #Optional. Passed regarding wether HFS constants A and B should be approximated for values of 0 above. NOT YET IMPLEMENTED
                     'ion_vel' : Float or [Float1,Float2] , #Velocity of particles, mono and bidirectional allowed. A single value can be passed and used for bidirectional splitting if bidir flag is set.
                     'bidir' : True , #Bool as to whether ion motion generates bidirection doppler shift
                     'PsiHFS' : True , #Bool, testing flag left in for completeness
                     'DoLowSig' : True , #Bool for whether Lowfield approximation and signal are included and plotted
                     'DoHighSig' : True , #REQUIRES TERM ENERGY!Bool for whether Highfield approximation and signal are included and plotted
                     'EtermG': 169086.9076, #Term energy of lower state in cm^-1 - Required for DoHighSig
                     'EtermE' : 191444.47832914, #Term energy of upper state in cm^-1 - Required for DoHighSig
                     'sortE' : True , #Bool as to whether the wavefunctions will attempt to be sorted based on expected resulting energy before diagonalization. This significantly slows down operations. 
                      }

    Returns
    -------
    data : Dictionary
        DESCRIPTION.

    '''

    # Define local variables from the inputdeck
    do_low = _approximation_enabled(Inputdeck.get('DoLowSig', False))
    do_high = _approximation_enabled(Inputdeck.get('DoHighSig', False))
    if 'I_spin' in Inputdeck and (do_low or do_high):
        raise NotImplementedError('HFS approximation spectra are not implemented; omit or disable DoLowSig/DoHighSig')

    S_ground = Inputdeck['s_ground']
    S_excited = Inputdeck['s_excited']
    L_ground = Inputdeck['l_ground']
    L_excited = Inputdeck['l_excited']
    E_ground = Inputdeck['E_ground']
    E_excited = Inputdeck['E_excited']
    Bfield_ext = Inputdeck['Bmag']
    Bext_angle = Inputdeck['b_angle']


    # Set polarization filter angle to 0 and define no polarization filter.
    Polangle = 0
    Polfilter = False
    if 'Pol_angle' in Inputdeck:  # Check if the inputdeck has a pol_angle and adjust parameters accordingly.
        Polangle = Inputdeck['Pol_angle']
        Polfilter = True


    if 'TAlpha' in Inputdeck:
        Talpha = Inputdeck['TAlpha']
    else:
        Talpha = 1
    
    if 'TBeta' in Inputdeck:
        Tbeta = Inputdeck['TBeta']
    else:
        Tbeta = 1
    
    if 'bidir' in Inputdeck:
        bidirect = Inputdeck['bidir']
    else:
        bidirect = False

    if 'sortE' in Inputdeck:
        sortE = Inputdeck['sortE']
    else:
        sortE = False

    if 'amu' in Inputdeck:
        pass
    else:
        Inputdeck['amu'] = 1  # Setting a default value so it's not required to be input.
     

    # 1. Establish the "Spectrometer" fallback range
    spec_data = Inputdeck.get('SpectrumData')
    # If data exists, use its bounds; otherwise, use the global default
    data_range = [np.min(spec_data[0]), np.max(spec_data[0])] if spec_data is not None else [1, 1000]
    
    # 2. Resolve windows: Dictionary Value > Spectrometer range > default
    specwindow = Inputdeck.get('spec_window', data_range)
    plotwindow = Inputdeck.get('plot_window', data_range)
    
    
    if 'EtermG' in Inputdeck:
        EtermG = Inputdeck['EtermG']
    else:
        EtermG = None
        
    if 'EtermE' in Inputdeck:
        EtermE = Inputdeck['EtermE']
    else:
        EtermE = None    
        
        
    if 'I_spin' in Inputdeck:  # Check if the input deck has nuclear spin as to calculate HFS splitting and zeeman effect or not.
        HFS_ConstsG = {}
        HFS_ConstsG['HFSA'] = Inputdeck['HFS_G']
        HFS_ConstsE = {}
        HFS_ConstsE['HFSA'] = Inputdeck['HFS_E']
        if 'HFSB_G' in Inputdeck:  # Check if either of the B HFS constants are in the input deck, otherwise make a copy of the A coefficients as zeros.
            HFS_ConstsG['HFSB'] = Inputdeck['HFSB_G']
        else:
            HFS_ConstsG['HFSB'] = np.zeros_like(E_ground)

        if 'HFSB_E' in Inputdeck:  # Check if either of the B HFS constants are in the input deck, otherwise make a copy of the A coefficients as zeros.
            HFS_ConstsE['HFSB'] = Inputdeck['HFSB_E']
        else:
            HFS_ConstsE['HFSB'] = np.zeros_like(E_excited)

        if 'PsiHFS' in Inputdeck:
            PsiHFS = Inputdeck['PsiHFS']
        else:
            PsiHFS = False
            

        # Use spin, spin parity, and nuclear dipole moment to calculate lande g_i nuclear factor. Might be off by a factor of 1836?
        I_spin = Inputdeck['I_spin']
        mu_I = Inputdeck['mu_I']
        thisg_I = lande_gi(I_spin, mu_I)

        # Process the wavefunction matrix and energies. Preliminary sorting is done so that the final state order should be roughly as expected.
        Ground_Level_HFS = stateprocess_HFS(L_ground, S_ground, I_spin, HFS_ConstsG, thisg_I, E_ground, Bfield_ext)
        Excited_Level_HFS = stateprocess_HFS(L_excited, S_excited, I_spin, HFS_ConstsE, thisg_I, E_excited, Bfield_ext)

        # Take the sorted states, the state matrix, and energy matrix and calculate the zeeman splitting (with HFS)
        Ground_Zeeman_HFS = Zeeman_func_HFS(Ground_Level_HFS, Bfield_ext, thisg_I, PsiHFS)
        Excited_Zeeman_HFS = Zeeman_func_HFS(Excited_Level_HFS, Bfield_ext, thisg_I, PsiHFS)
        # Use the eigenvectors and values from the diagonalization of the hamiltonian to calculate the relative signal strength. This function uses a combination of Isler1997 and Metcalf 4.32/4.33
        Z_sig_HFS = Zeeman_signal_HFS(Ground_Level_HFS, Ground_Zeeman_HFS, Excited_Level_HFS, Excited_Zeeman_HFS, Bangle=Bext_angle, gamma=Polangle, Filter=Polfilter,T_Alpha = Talpha, T_Beta = Tbeta)

        # Output a bunch of data
        data = {}
        data['signal'] = Z_sig_HFS[0]
        data['wave_vac'] = Z_sig_HFS[6]
        data['wave_air'] = Z_sig_HFS[1]
        data['HFSG_Energy'] = Ground_Level_HFS[2]
        data['HFSE_Energy'] = Excited_Level_HFS[2]
        data['statelist_Ground'] = Ground_Level_HFS[-1]
        data['statelist_Excited'] = Excited_Level_HFS[-1]
        data['RedDipoleOp'] = Z_sig_HFS[-3]  # This is the tensor/kronecker product of the upper and lower state matrices after the field has been included. afcf in calculations. Might not be what is expected.
        data['w3coeffs_ground'] = Ground_Level_HFS[0]
        data['w3coeffs_excited'] = Excited_Level_HFS[0]
        data['ZeemanSplits_G'] = Ground_Zeeman_HFS[-1]
        data['Red_Dipole'] = Z_sig_HFS[4]

        data['rad'] = Z_sig_HFS[2]  # Given out so that it can be used to plot specific polarizations.

    # If there is no nuclear spin in the input deck, do a similar calculation but no hpyerfine splitting ins included nor nuclear zeeman splitting.
    else:
        # Process the wavefunction matrix and energies. Preliminary sorting is done so that the final state order should be roughly as expected.
        Ground_Level = stateprocess(L_ground, S_ground, E_ground, Bfield_ext)
        Excited_Level = stateprocess(L_excited, S_excited, E_excited, Bfield_ext)
        data = {}
        # Preliminary workflow for attempting stark shift calculation. Presently doesn't provide reasonable results.

        # Use ground and excited states to calculate zeeman splitting and diangonalize the energy hamiltonian to obtain eigenvalues and vectors.
        Ground_Zeeman = Zeeman_func(Ground_Level, Bfield_ext)
        Excited_Zeeman = Zeeman_func(Excited_Level, Bfield_ext)
        # Use the new eigenvectors as the basis to calculate the relative signal strengths for the transitions
        Z_sig = Zeeman_signal(Ground_Level, Ground_Zeeman, Excited_Level, Excited_Zeeman, Bangle=Bext_angle, gamma=Polangle, Filter=Polfilter,T_Alpha = Talpha, T_Beta = Tbeta)
        Z_sig_norad = Zeeman_sig_norad(Ground_Level, Ground_Zeeman, Excited_Level, Excited_Zeeman)

        if do_low:
            Ground_Level = stateprocess(L_ground, S_ground, E_ground, Bfield_ext, sortE=sortE)
            Excited_Level = stateprocess(L_excited, S_excited, E_excited, Bfield_ext, sortE=sortE)
            Ground_Zeeman_Low = Zeeman_func(Ground_Level, Bfield_ext, Lowfield=True)
            Excited_Zeeman_Low = Zeeman_func(Excited_Level, Bfield_ext, Lowfield=True)
            Z_sig_Low = Zeeman_signal(Ground_Level, Ground_Zeeman_Low, Excited_Level, Excited_Zeeman_Low, Bangle=Bext_angle, gamma=Polangle, Filter=Polfilter,T_Alpha = Talpha, T_Beta = Tbeta)
            # Z_sig_Low = Zeemansig_Low(Ground_Level,Excited_Level,Bangle=Bext_angle,gamma=Polangle, Filter=Polfilter)
            data['signal_low'] = Z_sig_Low[0]
            data['wave_vac_low'] = Z_sig_Low[6]
            data['wave_air_low'] = Z_sig_Low[1]
            data['EigGround_low'] = Ground_Zeeman_Low
            data['EigExcited_low'] = Excited_Zeeman_Low
        if do_high:
            Ground_Level = stateprocess(L_ground, S_ground, E_ground, Bfield_ext, sortE=sortE)
            Excited_Level = stateprocess(L_excited, S_excited, E_excited, Bfield_ext, sortE=sortE)
            Ground_Zeeman_High = Zeeman_func(Ground_Level, Bfield_ext, Highfield=True, Eterm=EtermG)
            Excited_Zeeman_High = Zeeman_func(Excited_Level, Bfield_ext, Highfield=True,Eterm=EtermE)
            Z_sig_High = _zeeman_signal(Ground_Level, Ground_Zeeman_High, Excited_Level, Excited_Zeeman_High, Bangle=Bext_angle, gamma=Polangle, Filter=Polfilter,T_Alpha = Talpha, T_Beta = Tbeta, label_eigenpairs=False)
            # Z_sig_Low = Zeemansig_Low(Ground_Level,Excited_Level,Bangle=Bext_angle,gamma=Polangle, Filter=Polfilter)

            data['signal_high'] = Z_sig_High[0]
            data['wave_vac_high'] = Z_sig_High[6]
            data['wave_air_high'] = Z_sig_High[1]
            data['EigGround_high'] = Ground_Zeeman_High
            data['EigExcited_high'] = Excited_Zeeman_High
        data['EigGround'] = Ground_Zeeman
        data['EigExcited'] = Excited_Zeeman
        data['ZeemanSplits_G'] = Ground_Zeeman[-1]


        data['sigsum'] = Z_sig[7]
        data['SigNotNorm'] = Z_sig[8]
        data['bigstatelist'] = Z_sig[3]
        data['signal'] = Z_sig[0]
        data['wave_vac'] = Z_sig[6]
        data['wave_air'] = Z_sig[1]
        data['statelist_Ground'] = Ground_Level[-1]
        data['statelist_Excited'] = Excited_Level[-1]

        # Energy level output with lowfield approximation of zeeman splitting output.
        data['E0_Ground'] = Ground_Level[3]
        data['E0_Excited'] = Excited_Level[3]

        data['rad'] = Z_sig[2]
        data['Zsig_norad'] = Z_sig_norad
        # A_Ein isn't polarization normalized,
        data['A_Ein'] = [A_Ein(data['wave_air'][i], dips)/(2*data['bigstatelist'][i][1][0] + 1) for i, dips in enumerate(data['signal'])]
        data['A_Ein_polavg'] = [2*A_Ein(data['wave_air'][i], dips)/(3*(2*data['bigstatelist'][i][1][0] + 1)) for i, dips in enumerate(data['Zsig_norad'])]
        
        data['sigavgs'] = [sigs/(2*data['bigstatelist'][i][1][0] +1) for i,sigs in enumerate(data['signal']) ]

        data['f_osc'] = np.sum(data['signal'])*m_elec/(4*hbar*(np.pi**3))*((2*L_ground+1)/(2*L_excited+1))
        data['S_line'] = np.sqrt(np.sum([(3 /(16*q_charge*np.pi**4*nm_to_Hz(x)))*data['signal'][i] for i,x in enumerate(data['wave_vac'])]))*((L_ground+1)/(2*L_excited+1))
        data['afcf'] = Z_sig[4]

    # There are several instrument functions available for convolution. Lorrentzian, Skewed Lorrentzian, Doppler, and DopplerInstrum, which is a combination of a doppler and instrument (not a Pseudo Voigt though.)

    # These conditionals can very much be cleaned up and simplified in the future.

    if 'ion_vel' in Inputdeck:
        ion_vel = Inputdeck['ion_vel']

    else:
        ion_vel = 0
    if 'Convfxn' in Inputdeck:
        Convtype = Inputdeck['Convfxn']
        convtemp = 300
        SkewedConstants = [1, 1]
        if 'specstep' in Inputdeck:
            stepsize = Inputdeck['specstep']
        else:
            stepsize = 0.002 #Set a default step size 
        if Inputdeck['Convfxn'] == 'Gaussian' or Inputdeck['Convfxn'] == 'GaussianInstrum':
            #Requires Input Temperature for non Lorrentzian lineshapes
            if 'TempeV' in Inputdeck: #Allow for temp to be input in eV as well.
                convtemp = Inputdeck['TempeV']*11602 
            else:               
                convtemp = Inputdeck['Temp']
                
            SkewedConstants = [1, 1]

        if Inputdeck['Convfxn'] == 'Skewed' or Inputdeck['Convfxn'] == 'Lorrentzian':

            if 'Skewness' in Inputdeck:
                SkewedConstants = Inputdeck['Skewness']
            else:
                SkewedConstants = [1, 1]
        if Inputdeck['Convfxn'] == 'Voigt':
            if 'TempeV' in Inputdeck: #Allow for temp to be input in eV as well.
                convtemp = Inputdeck['TempeV']*11602 
            else:               
                convtemp = Inputdeck['Temp']

        if 'fxnwindow' in Inputdeck:
            fxnwindow = Inputdeck['fxnwindow']
        else:
            fxnwindow = 0.1  # This is how far the bin window will extend beyond the edge of the window. May need to be extended if large broadening.
    else:
        # Setting the convolution window for the function

        # This is the default configuration using a normal Lorrentzian lineshape.
        Convtype = 'Lorrentzian'
        SkewedConstants = [1, 1]
        convtemp = 300
        ion_vel = 0
        if 'fxnwindow' in Inputdeck:
            fxnwindow = Inputdeck['fxnwindow']
        else:
            fxnwindow = 0.1  # This is how far the bin window will extend beyond the edge of the window. May need to be extended if large broadening.
        if 'specstep' in Inputdeck:

            stepsize = Inputdeck['specstep']
        else:
            stepsize = 0.002222  # This is the resolution of the spectrometer pixels in nm.

    if 'specres' in Inputdeck:
        specres = Inputdeck['specres']
    else:
        specres = 10
        
    #Make the convolution based on the sticks.
    ConvolvedSpect, redsticks, binwinds = Convol_Spect(data['wave_air'], data['signal'],
                                                       specwindow, plotwindow, Inputdeck['amu'], stepsize,
                                                       Temperature_in=convtemp, functiontype=Convtype,
                                                       wind_size=fxnwindow, Skewness_consts=SkewedConstants, ionvel=ion_vel, specres=specres, bidir=bidirect)
    
    data['SpecOut'] = ConvolvedSpect
    data['reduced_sticks'] = redsticks
    data['bin_windows'] = binwinds

    pilist = np.array([x[1] for x in data['rad']]) == 0
    sigmaplus = np.array([x[1] for x in data['rad']]) == -1  # The sign is flipped.
    sigmaminus = np.array([x[1] for x in data['rad']]) == 1  # If you start from the bottom, you have the change the sign here
    otherlists = np.array([x[1] for x in data['rad']]) == 3

    # data['radlists'] = np.array([pilist,sigmaplus,sigmaminus])

    # data['radlists'] = np.array([pilist,sigmaplus,sigmaminus])
    data['pi_sticks'] = [data['wave_air'][pilist], data['signal'][pilist]]
    data['sigmaplus_sticks'] = [data['wave_air'][sigmaplus], data['signal'][sigmaplus]]
    data['sigmaminus_sticks'] = [data['wave_air'][sigmaminus], data['signal'][sigmaminus]]
    data['other_sticks'] = [data['wave_air'][otherlists], data['signal'][otherlists]]

    if do_low:
        ConvolvedSpectLow, redsticksLow, binwindsLow = Convol_Spect(data['wave_air_low'], data['signal_low'],
                                                                    specwindow, plotwindow,  Inputdeck['amu'], stepsize,
                                                                    Temperature_in=convtemp, functiontype=Convtype,
                                                                    wind_size=fxnwindow, Skewness_consts=SkewedConstants, ionvel=ion_vel, specres=specres, bidir=bidirect)

        data['SpecOutLow'] = np.array(ConvolvedSpectLow) #Ugly, but should fix an issue with plotting it?
        data['reduced_sticksLow'] = redsticksLow
        # data['bin_windows'] = binwindsLow

    if do_high:
        ConvolvedSpectHigh, redsticksHigh, binwindsHigh = Convol_Spect(data['wave_air_high'], data['signal_high'],
                                                                    specwindow, plotwindow,  Inputdeck['amu'], stepsize,
                                                                    Temperature_in=convtemp, functiontype=Convtype,
                                                                    wind_size=fxnwindow, Skewness_consts=SkewedConstants, ionvel=ion_vel, specres=specres, bidir=bidirect)

        data['SpecOutHigh'] = np.array(ConvolvedSpectHigh) #Ugly, but it should fix an issue with plotting it?
        data['reduced_sticksHigh'] = redsticksHigh
        # data['bin_windows'] = binwindsLow
    data['Input'] = Inputdeck
    return data

