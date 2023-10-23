from numpy.typing import ArrayLike
import scipy.constants as const
import numpy as np
from matplotlib import pyplot as plt
from .utils.logging import setup_log
logger = setup_log(__name__)

class Spectrum(object):
    
    def __init__(self, wavel: ArrayLike, fluxes: ArrayLike):
        logger.info('Using nm and Jy as units')
        self._l = wavel #Angstroms
        self._f = fluxes #for now unit-less normalized flux
        #self._parameters = parameters #dictionary of parameters describing spectrum
    
    def __call__(self):
        # ToDo: spline to evaluate flux at each wavelength?
        return self.spectrum.T
    
    @property
    def spectrum(self):
        return np.array([self._l,self._f])
    
    @property
    def wavel(self):
        #return wavelengths in Angstroms
        return self._l
    
    @property
    def freq(self):
        #returns frequency in Hz
        return const.speed_of_light / (self._l*1e-10)
    
    @property
    def flux(self):
        #returns normalized fluxes
        return self._f
    
    def plot(self,ax=None,**kwargs):
        if ax is None:
            
            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()
            
        ax.plot(self.wavel,self.flux,**kwargs)
        ax.set_xlabel(r'$\lambda [A]$')
        ax.set_ylabel(r'$\hat{n}$')

        return fig

class ObservedSpectrum(Spectrum):
    
    def __init__():
        super(Spectrum, self).__init__()
    
    def from_csv():
        pass
    
class SimulatedSpectrum():
    """ToDo:
    - binning
    - reddening
    - convolution
    """
    def __init__():
        super(ObservedSpectrum, self).__init__()