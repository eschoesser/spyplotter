from numpy.typing import ArrayLike
from typing import List
import scipy.constants as const
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from pathlib import Path

from .powr import readWRPlotDatasets
from .utils.logging import setup_log
logger = setup_log(__name__)

class Spectrum(object):
    
    def __init__(self, x: ArrayLike, y: ArrayLike, x_unit:u.Unit=None, y_unit:u.Unit=None):
        """Generic class for a spectrum. The same class is used for observed and model spectra

        :param x: array of wavelength or frequency values
        :type x: ArrayLike
        :param y: flux, default: normalized spectrum
        :type y: ArrayLike
        :param x_unit: unit of the x-axis of the spectrum, default is nanometer
        :type x_unit: u.Unit, optional
        :param y_unit: flux unit of the spectrum, default is None = normalized spectrum
        :type y_unit: u.Unit, optional
        """
        #ToDo: dictionary of parameters, maybe a string with name of the spectrum
        if type(x) == u.quantity.Quantity:
            # if x has already unit, don't change it
            logger.info(f'Keeping units of x: {x.unit}')
            self._x = x
        elif x_unit == None:
            # if no unit is given, use nm
            logger.info('As no unit for x was specified, Angstrom are assumed')
            self._x = x * u.AA
        else:
            # if unit is specified and x does not have unit, take specified unit
            self._x = x * x_unit
        
        if type(y) == u.quantity.Quantity:
            # if y has already unit, don't change it
            logger.info(f'Keeping units of y: {y.unit}')
            self._y = y
            
            #if pre-specified unit is dimensionless, assume a normalized spectrum
            if y.unit == u.dimensionless_unscaled:
                self._normalized = True
            else:
                self.normalized = False
                
        elif y_unit == None:
            #if y is not an astropy quantity and no unit was specified, assume a normalized spectrum
            logger.info('As no unit for y was given, a normalized spectrum is assumed')
            self._y = y * u.dimensionless_unscaled
            self._normalized = True
        else:
            # if unit is specified and y does not have unit, take specified unit
            self._y = y * y_unit
            self._normalized = False
        
    
    def __call__(self):
        # ToDo: when called at a specific x point, use spline or interpolation to evaluate flux at given point?
        return self.spectrum.T
    
    @property
    def spectrum(self) -> ArrayLike:
        """Convert spectrum in a 2D numpy array 

        :return: spectrum as numpy array
        :rtype: ArrayLike
        """
        return np.array([self._x,self._y])
    
    @property
    def x(self,unit:u.Unit=None) -> u.quantity.Quantity:
        """Return x values of spectrum

        :param unit: unit of returned value, defaults to None
        :type unit: u.Unit, optional
        :return: _description_
        :rtype: _type_
        """
        if unit == None:
            return self._x
        else:
            return self._x.to(unit,equivalencies=u.spectral())
    
    @property
    def y(self, unit:u.Unit=None) -> u.quantity.Quantity:
        if unit == None:
            return self._y
        else:
            return self._y.to(unit,equivalencies=u.spectral())
        
    @property
    def x_unit(self) -> u.Unit:
        return self._x.unit
    
    @property
    def y_unit(self) -> u.Unit:
        return self._y.unit
    
    @property
    def normalized(self) -> bool:
        return self._normalized
        
    @classmethod
    def from_powr(cls, filepath:str, keywords:List[int]=[''], dataset:int=1, xunit:u.Unit=None,yunit:u.Unit=None):
        #ToDo: Check if all key words correspond to normalized or unnormalized spectrum
        
        path = Path(filepath)
        
        #Check if model path exists
        if path.exists():
            
            if path.is_file():
                x,y = readWRPlotDatasets(filepath,keywords,dataset)
                
            elif (path / 'formal.plot').exists():
                logger.info('Reading formal.plot')
                x,y = readWRPlotDatasets(path / 'formal.plot',keywords,dataset)
        else:
            logger.error('Path does not exist')
            raise ValueError
        
        if yunit is None:
            #search for signs of a flux calibrated spectrum, otherwise assume normalized spectrum
            if (('CONTINUUM' in keywords[0]) or ('EMERGENT' in keywords[0])) or dataset > 1:
                yunit = u.erg/u.s/u.cm**2/u.AA
                logger.info(f'Flux calibrated spectrum at 10 pc. Thus using {yunit} as y unit.')
            else:
                logger.info('No flux unit specified and no signs for y units detected. Thus assuming normalized spectum.')
                yunit = None
            
        return cls(x,y,xunit,yunit)
    
    @classmethod
    def from_file(cls,filename:str, skiprows:int=0, delimiter:str=' ',xunit:u.Unit=None,yunit:u.Unit=None,**kwargs):
        path = Path(filename)
        if path.exists():
            data = np.loadtxt(filename, skiprows=skiprows,delimiter=delimiter,**kwargs)
        else:
            logger.error('Path does not exist')
            raise ValueError
        
        return cls(data[:,0], data[:,1], xunit, yunit)
        
    
    @classmethod
    def from_cmfgen(cls,filepath:str):
        #todo: write a function that imports from a CMFGEN output file a specific simulated Spectrum class
        pass
    
    def to_file(self,filename:str):
        #todo: write a function that saves table of current spectrum in a csv file
        #advanced todo: save also all parameters that were specified in header
        pass
    
    def convert_units(self,x_unit: u.Unit=None,y_unit: u.Unit =None):
        
        if (x_unit is None) and (y_unit is None):
            logger.info('Specify which (x_unit or y_unit) should be converted and specify the desired unit using astropy.units')
            pass
        
        if x_unit is not None:
            logger.debug(f'The unit of all x values is changed to {x_unit}')
            self._x = self._x.to(x_unit,equivalencies=u.spectral())
            
        if y_unit is not None:
            logger.debug(f'The unit of all y values is changed to {y_unit}')
            self._x = self._y.to(y_unit,equivalencies=u.spectral())
     
    def plot(self,x_unit:u.Unit=None, y_unit:u.Unit=None, interval:ArrayLike=None, ax=None,**kwargs):
        """Function to plot the spectrum
        The units can be changed for the plotting. However, 
        the units are only changed for the plot and not the whole class
        
        If you would like to convert the units permanently, use the 
        convert_to() function.

        :param x_unit: unit of x-values, defaults to None
        :type x_unit: u.Unit, optional
        :param y_unit: unit of y-values, defaults to None
        :type y_unit: u.Unit, optional
        :param ax: can be specified to plot in a previously created figure, defaults to None
        :type ax: Axis, optional
        :return: Figure
        """
        
        #Convert to different 
        #does not convert whole spectrum - only for plotting
        if x_unit is None: 
            logger.debug(f'Using the following pre-specified units:\n\tx:{self._x.unit}')
            x = self._x
        else:
            logger.debug(f'Using following new units:\n\tx: {x_unit}')
            x = self._x.to(x_unit,equivalencies=u.spectral(),)
            
        if y_unit is None: 
            logger.debug(f'Using the following pre-specified units:\n\ty:{self._y.unit}')
            y = self._y
        else:
            logger.debug(f'Using following new units:\n\ty: {y_unit}')
            y = self._y.to(y_unit,equivalencies=u.spectral())
            
        if ax is None:
            
            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()
            
        ax.plot(x,y,**kwargs)
        
        #Change labels of x axis dependent on unit type
        if x.unit.physical_type == u.m.physical_type:
            logger.debug('wavelength like units used')
            #wavelength type unit
            ax.set_xlabel(r'$\lambda$' + f' [{x.unit:latex}]' )
        elif x.unit.physical_type == u.Hz.physical_type:
            logger.debug('frequency like units used')
            #frequency type unit
            ax.set_xlabel(r'$\nu$' + f' [{x.unit:latex}]')
        elif x.unit.physical_type == u.J.physical_type:
            logger.debug('energy like units used')
            #energy type unit
            ax.set_xlabel(r'$E$'+f' [{x.unit:latex}]')
        
        #Change labels of y axis if it is a normalized spectrum
        if self._normalized:
            ax.set_ylabel(r'$\hat{n}$')
        else:
            ax.set_ylabel(f' Flux [{y.unit:latex}]')
            
        if interval is not None:
            ax.set_xlim(interval[0],interval[1])

        return fig
    
    def zoom_plot(self,intervals:ArrayLike,fig_width=10,fig_height=4, x_unit:u.Unit=None, y_unit:u.Unit=None,**kwargs):
        #ToDo: Add option for specifying multiple columns
        
        n_int = len(intervals)
        
        fig, ax = plt.subplots(n_int,1,figsize=(fig_width,fig_height*n_int))
        
        for i in range(n_int):
            self.plot(x_unit, y_unit, interval=intervals[i],ax=ax[i],**kwargs)
            
        return fig
    
    