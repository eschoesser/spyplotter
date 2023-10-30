from .spectrum import Spectrum

class Model:
    '''
    should contain all information that is within a model
    In case of PoWR:
    - parameters read from kasdef
    - all stellar structures that were plotted (maybe as dictionary containing all structures pointing to own class) - velocity structure, temperature structure, ionization levels...
    - all spectra read from formal.plot saved in dictionary
    - functions to save state of model to hdf5 (or fits) file and read again 
    '''
    
    def __init__(self, params:dict,spectrum:Spectrum, ):
        
        self._params = params
        
        self._spectrum = spectrum
        
    @classmethod
    def from_powr(cls):
        pass
        
    @classmethod
    def from_cmfgen(cls):
        pass    
    