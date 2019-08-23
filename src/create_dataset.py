"""Create csv with spectral data"""
from os import getcwd
from pathlib import Path

from astropy.io import fits
import pandas as pd

PROJECT_PATH = getcwd()
SPECTRA = {}
for spectrum_path in Path('%s/data/fits/' % PROJECT_PATH).glob('*fits'):
    spectrum_fits = fits.open(spectrum_path)
    spectrum = spectrum_fits[1].data[0]
    SPECTRA[spectrum_fits[0].header['OBJID']] = spectrum
    Path(spectrum_path).unlink()
wavelenght = spectrum_fits[4].data[0]

all_spectra = pd.DataFrame(SPECTRA, index=wavelenght).T

all_spectra.to_csv('%s/data/all_spectra.csv' % PROJECT_PATH)

Path(PROJECT_PATH + '/models').mkdir(exist_ok=True)
