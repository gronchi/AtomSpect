"""Reading and saving spectra and input decks."""
import csv

import numpy as np

from .lineshapes import Normalize







def read_Spectra(filename, scalewave=1, headercount=1, keepheaders=False):
    '''
    read_Spectra is a general spectra import that is flexible enough for multiple columns of spectrometer data (Say for multiple field strengths)
    The default assumption is that the wavelengths are in nm, but you can set scalewave to adjust as you input if needed.
    If your data file has headers with details you wish to keep correlated, the keepheaders flag will keep all headers as an output.

    Parameters
    ----------
    filename : string/path
        Destination/name of the file to be imported
    scalewave : float, optional
        Scaling for wavelengths in should it be desired to quickly convert from A to nm or etc.The default is 1.
    headercount : float, optional
        How many header rows are in the data.The default is 1.
    keepheaders : BOOL, optional
        Flag for whether the data within the header of the file is kept and returned for the user.The default is False.

    Returns
    -------
    specdata_out : TYPE
        If the header data is not kept:
            Output is [wavelengths,signals]
        If headers are kept, output is [[wavelengths,signals],[headers]]

    '''
    with open(f'{filename}', mode='r', newline='') as readme:
        reader = csv.reader(readme, delimiter=',', quoting=csv.QUOTE_NONE)
        headers = []
        for i in range(headercount):
            headers.append(next(reader))
        specdat = []

        # Read through the rows and scale the wavelength by the scalefactor.
        for rows in reader:
            specdat.append([float(x) for x in rows])

        specdat = np.array(specdat).T  # Transpose so that the ith entry is all wavelengths or intensities.
        specwav = specdat[0]

        Spectraout = []
        [Spectraout.append(x) for i, x in enumerate(specdat[1:])]

        # This small bit gives magnetic field values based on the headers info
        # Bvals = [float(x) for x in headers[0][1:]]

        specdata_out = {}
        specdata_out['wavelength'] = [scalewave*x for x in specwav]  # output wavelength, assumed in nm.
        # specdata_out['Bvals'] = Bvals
        if len(Spectraout) == 1:
            specdata_out['signals'] = Spectraout[0]
        else:
            specdata_out['signals'] = Spectraout

    if keepheaders == True:
        return [specdata_out, headers]
    else:
        return specdata_out


def read_dict(filename):

    with open(f'{filename}', mode='r', newline='') as readme:
        reader = csv.DictReader(readme, delimiter=',', quoting=csv.QUOTE_NONE)
        # headers = []
        # for i in range(headercount):
        #     headers.append(next(reader))
        datraw = []

        # Read through the rows and scale the wavelength by the scalefactor.
        for rows in reader:
            trow = {k: v for k, v in rows.items() if v}

            for keys in trow.keys():
                try:
                    trow[keys] = float(trow[keys])
                except:
                    pass

            datraw.append(trow)

    return datraw

def read_HFSInput(filename):

    with open(f'{filename}', mode='r', newline='') as readme:
        reader = csv.DictReader(readme, delimiter=',', quoting=csv.QUOTE_NONE)
        # headers = []
        # for i in range(headercount):
        #     headers.append(next(reader))
        datraw = []

        # Read through the rows and scale the wavelength by the scalefactor.
        for rows in reader:
            trow = {k: v for k, v in rows.items() if v}

            for keys in trow.keys():
                try:
                    trow[keys] = float(trow[keys])
                except:
                    pass

            datraw.append(trow)

    return datraw
#%% Save and Load
def Savenpy_Dict(filename, Dict_in,spectra = None, Normalize_sig = False):
    #This function should allow for saving of data so it doesn't need to be re-input everytime.
    #This includes the input dictionary, which could then be easily compiled and re-called later.
    #Work on this has not been fleshed out and it should be treated as such.
    savedat = {}

    if spectra:
        specdat = {}
        try: #Try assuming spectra is a dictionary with keys['wavelength' and 'signals'
           specdat['wavelength'] = np.array(spectra['wavelength'] )
        except: #Else, spectra should be [wavelength,signals] , with each as arrays. 
            specdat['wavelength'] = np.array(spectra[0] )
        #Allow for signals to be multiple lists
        try :
            len(np.shape(spectra['signals']))
            if len(np.shape(spectra['signals']))==2:
                if Normalize_sig:
                    specdat['signals'] = [Normalize(x) for x in spectra['signals']] 
                else:
                    specdat['signals'] = [x for x in spectra['signals']] 

            else:
                if Normalize_sig:

                    specdat['signals'] = Normalize(spectra['signals'])
                else:
                    specdat['signals'] = spectra['signals']

        except:
            if len(np.shape(spectra[1]))==2:
                if Normalize_sig:
                    specdat['signals'] = [Normalize(x) for x in spectra[1]] 
                else:
                    specdat['signals'] = [x for x in spectra[1]] 

            else:
                if Normalize_sig:
                    specdat['signals'] = Normalize(spectra[1])
                else:
                    specdat['signals'] = spectra[1]

  
    
        savedat['Specdata'] = specdat
    savedat['data'] = Dict_in
    np.save(f'{filename}',savedat,allow_pickle=True)
    return

def Loadnpy_Dict(filepath):
    #This function takes the numpy.save dictionary and loads it, including the spectrometer data, for future use.
    #Need to test if figures can be saved too...
    thisload = np.load(f'{filepath}.npy', allow_pickle=True)
    loadeddict = thisload.item().get('data')
    try:
        loadedspec = thisload.item().get('Specdata')
        return loadeddict, loadedspec
    except:
        pass
        # print('Failed to load Spectrometer data - May not exist')
    return loadeddict
