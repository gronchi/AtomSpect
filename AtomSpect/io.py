"""Reading and saving spectra and input decks."""
import csv
from pathlib import Path

import numpy as np

from .lineshapes import Normalize







def read_Spectra(filename, scalewave=1, headercount=1, keepheaders=False):
    """Read a CSV whose first column is wavelength and remaining columns are signals.

    filename is a path; headercount is the number of leading rows to skip.
    scalewave multiplies wavelengths (use 0.1 for angstroms to nm).
    Return a dictionary with wavelength and signals. A single signal column
    is an array; multiple columns form a list of arrays. If keepheaders=True,
    return [dictionary, header_rows]. All data cells must be numeric."""
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

    """Read CSV records, omit empty cells, and convert numeric cells to floats."""
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
                except (ValueError, TypeError):
                    pass

            datraw.append(trow)

    return datraw

def read_HFSInput(filename):

    """Read hyperfine CSV records using the same conversion rules as read_dict."""
    return read_dict(filename)
#%% Save and Load
def Savenpy_Dict(filename, Dict_in,spectra = None, Normalize_sig = False):
    #This function should allow for saving of data so it doesn't need to be re-input everytime.
    #This includes the input dictionary, which could then be easily compiled and re-called later.
    #Work on this has not been fleshed out and it should be treated as such.
    """Save a plain copy of Dict_in and optional spectra to a NumPy .npy file.
    
    spectra can be a dictionary with wavelength/signals or a two-row
    array-like [wavelength, signals]. Normalize_sig scales each signal to
    unit peak. NumPy appends .npy when filename does not already have it.
    The format uses pickle; load files only from trusted sources."""
    savedat = {'data': dict(Dict_in)}
    if spectra is not None:
        if isinstance(spectra, dict):
            wavelength, signals = spectra['wavelength'], spectra['signals']
        else:
            wavelength, signals = spectra[0], spectra[1]
        signal_array = np.asarray(signals)
        if Normalize_sig:
            signals = ([Normalize(row) for row in signal_array] if signal_array.ndim == 2
                       else Normalize(signal_array))
        savedat['Specdata'] = {'wavelength': np.asarray(wavelength), 'signals': signals}
    np.save(str(filename), savedat, allow_pickle=True)

def Loadnpy_Dict(filepath):
    #This function takes the numpy.save dictionary and loads it, including the spectrometer data, for future use.
    #Need to test if figures can be saved too...
    """Load a trusted file written by Savenpy_Dict.

    filepath may include or omit .npy. Return (plain_dictionary, spectra),
    with spectra=None if absent. This format enables pickle loading."""
    path = Path(filepath)
    if path.suffix != '.npy':
        path = Path(str(path) + '.npy')
    saved = np.load(path, allow_pickle=True).item()
    return saved.get('data'), saved.get('Specdata')
