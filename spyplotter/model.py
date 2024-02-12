import abc
from astropy import units as u
from typing import List
from pathlib import Path

from .spectrum import Spectrum
from .utils.logging import setup_log
from .powr import read_params_from_kasdefs
logger = setup_log(__name__)

class Model(object,metaclass=abc.ABCMeta):
    '''
    should contain all information about model
    - All methods are defined as abstract so that they need to be implemented dependent on model
    '''
    
    def __init__(self, directory_name):
        self._directory = directory_name
        
        self._spectrum = None
        
        self._params = None
    
    @property
    def directory(self):
        return self._directory
    
    @property
    def spectrum(self):
        
        if self._spectrum is None:
            logger.error('Spectrum was not read yet. Call read_spectrum first')
        else:
            return self._spectrum
    
    @property
    def params(self):
        
       if self._params is None:
            logger.error('Spectrum was not read yet. Call read_params first')
       else:
            return self._params
    
    @abc.abstractmethod
    def read_spectrum(self):
        #Implemented dependend on model type (see PoWRModel)
        NotImplementedError()
    
    @abc.abstractmethod
    def read_params(self):
        NotImplementedError()
    
    
class PoWRModel(Model):
    
    def __init__(self, directory_name):
        super().__init__(directory_name)
        
    def _check_path(self,file_path):
        #Check if file path contains the model directory 
        #If not: combine file name with directory
        if Path(file_path).parent != self._directory:
            new_dir = self._directory / Path(file_path)
        else: 
            new_dir = file_path
        if new_dir.exists():
            return new_dir
        else: 
            logger.error(f'{new_dir} does not exist')
        
    def read_spectrum(self, keywords:List=[''], filename='formal.plot',
                       dataset:int=1, xunit:u.Unit=None,yunit:u.Unit=None,
                       name=None,vrad=0. * u.km/u.s):
        
        spectrum_dir = self._check_path(filename)
        
        self._spectrum = Spectrum.from_powr(spectrum_dir, keywords=keywords, dataset=dataset,xunit=xunit,yunit=yunit,name=name,vrad=vrad)
        
    def read_params(self,filename='modinfo.kasdefs'):
        
        param_dir = self._check_path(filename)
            
        if filename == 'modinfo.kasdefs':
            self._params = read_params_from_kasdefs(param_dir)
    