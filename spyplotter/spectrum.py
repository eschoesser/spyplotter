from numpy.typing import ArrayLike
from typing import List
import scipy.constants as const
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SpectralCoord, SpectralQuantity
from pathlib import Path

from .powr import readWRPlotDatasets
from .utils.logging import setup_log
logger = setup_log(__name__)

class Spectrum(object):
    
    def __init__(self, x: ArrayLike, y: ArrayLike, x_unit:u.Unit=None, y_unit:u.Unit=None,name:str=None,vrad=0 * u.km / u.s):
        """Generic class for a spectrum. The same class is used for observed and model spectra

        :param x: array of wavelength or frequency values (if they have unit, keep unit)
        :type x: ArrayLike
        :param y: flux, default: normalized spectrum
        :type y: ArrayLike
        :param x_unit: unit of the x-axis of the spectrum, default is nanometer
        :type x_unit: u.Unit, optional
        :param y_unit: flux unit of the spectrum, default is None = normalized spectrum
        :type y_unit: u.Unit, optional
        :param name: optional name for the spectra, can be used for labelling them
        :type name: string
        :param vrad: radial velocity, if only float, use km/s
        :type vrad: float, SpectralCoord, SpectralQuantity or u.quantity.Quantity
        """
        self.name = name
        
        if isinstance(x,(u.quantity.Quantity,SpectralCoord,SpectralQuantity)):
            # if x has already unit, don't change it
            logger.info(f'Keeping units of x: {x.unit}')
            self._x = SpectralCoord(x)
        elif x_unit == None:
            # if no unit is given, use nm
            logger.info('As no unit for x was specified, Angstrom are assumed')
            self._x = SpectralCoord(x * u.AA)
        else:
            # if unit is specified and x does not have unit, take specified unit
            self._x = SpectralCoord(x * x_unit)
        
        if isinstance(y,(u.quantity.Quantity,SpectralCoord,SpectralQuantity)):
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
        
        v = self.check_velocity_unit(vrad)
        self._vrad = v
        self._x = self._x.with_radial_velocity_shift(v)
    
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
    def x(self,unit:u.Unit=None):
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
    def y(self, unit:u.Unit=None):
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
    
    @property
    def vrad(self):
        return self._vrad
        
    @classmethod
    def from_powr(cls, filepath, keywords:List[int]=[''], dataset:int=1, xunit:u.Unit=None,yunit:u.Unit=None,name=None,vrad=0. * u.km/u.s):
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
            
        return cls(x,y,xunit,yunit,name,vrad)
    
    @classmethod
    def from_file(cls,filename, skiprows:int=0, delimiter:str=' ',xunit:u.Unit=None,yunit:u.Unit=None,name:str=None,vrad=0.*u.km/u.s,**kwargs):
        """read spectrum from a file

        :param filename: path and file name
        :type filename: str
        :param skiprows: number of rows that are skipped before spectrum is read, defaults to 0
        :type skiprows: int, optional
        :param delimiter: delimiter between values in table of file, defaults to whitespace ' '
        :type delimiter: str, optional
        :param xunit: astropy unit of x, defaults to None
        :type xunit: u.Unit, optional
        :param yunit: y unit, defaults to None
        :type yunit: u.Unit, optional
        :param name: name of spectrum, defaults to None
        :type name: str, optional
        :raises ValueError: if path of given file does not exist
        :return: Spectrum object
        :rtype: Spectrum
        """
        path = Path(filename)
        if path.exists():
            data = np.loadtxt(filename, skiprows=skiprows,delimiter=delimiter,**kwargs)
        else:
            logger.error('Path does not exist')
            raise ValueError
        
        return cls(data[:,0], data[:,1], xunit, yunit,name,vrad)
    
    @classmethod
    def from_cmfgen(cls,filepath,name:str=None):
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
            
    def check_velocity_unit(self,v):
        
        if isinstance(v,(u.quantity.Quantity,SpectralCoord,SpectralQuantity)):
            logger.debug(f'Use given velocity unit: {v.unit}')
            return v
            
        elif isinstance(v,(float,int)):
            logger.info('No unit for vrad specified. Thus assuming km/s.')
            return v * u.km / u.s
        else:
            logger.error('Not known format for vrad used. Convert to float or astropy classes Quantity, SpectralCoord or SpectralQuantity')
            raise ValueError
        
    def doppler_shifted_x(self,vrad):
                
        #Check and set units of vrad
        v = self.check_velocity_unit(vrad)
        
        new_vrad = self._vrad + v
        
        #warn if spectrum was shifted before
        if self.vrad != 0:
            logger.warning(f'Spectrum was already shifted using vrad={self.vrad}. \nNow total shift of spectrum: vradtot = {self.vrad + vrad}')
        
        return self._x.with_radial_velocity_shift(new_vrad)
    
    def apply_shift_vrad(self,vrad,overwrite=True):
        """Apply radial shift to spectrum, overwrites current spectrum

        :param vrad: radial velocity
        :type vrad: float or astropy classes Quantity, SpectralCoord or SpectralQuantity
        :raises ValueError: if vrad has not one of required formats
        """
        
        if overwrite:
            #Overwrite spectrum 
            logger.debug('Overwrite spectrum when applying velocity shift')
            
            self._x = self.doppler_shifted_x(vrad)
            self._vrad = vrad + self.vrad
            
            return self._x
        else:
            
            #Return new object of spectrum
            logger.debug('Return new object with shifted x-values')
            
            #check and set velocity unit
            v = self.check_velocity_unit(vrad)
            
            new_vrad = self._vrad + v
            
            return Spectrum(self.x,self.y,self.x_unit,self.y_unit,self.name,new_vrad)
    
    def to_velocity_space(self,x_rest,doppler_convention='optical',v_unit=u.km/u.s,vrad=0. * u.km/u.s):
        
        if vrad != 0:
            x = self.doppler_shifted_x(vrad)
        else:
            x = self._x
            
        if isinstance(x_rest,(float,int)):
            x_rest = x_rest * u.AA
            logger.info('No units for vrad specified. Thus using Angstrom')
            
        return x.to(v_unit,doppler_convention=doppler_convention,doppler_rest=x_rest)
    
    def plot_velocity(self, x_rest, doppler_convention='optical', v_unit=u.km/u.s, vrad=0. * u.km/u.s,
                      ax=None, fig_width=10, fig_height=4, y_unit:u.Unit=None, interval:ArrayLike=None, **kwargs):
        
        v = self.to_velocity_space(x_rest=x_rest,doppler_convention=doppler_convention,v_unit=v_unit,vrad=vrad)
        
        if ax is None:
            
            fig, ax = plt.subplots(figsize=(fig_width,fig_height))

        else:

            fig = ax.get_figure()
        
        #changing y unit if specified
        if y_unit is None: 
            logger.debug(f'Using the following pre-specified units:\n\ty:{self._y.unit}')
            y = self._y
        else:
            logger.debug(f'Using following new units:\n\ty: {y_unit}')
            y = self._y.to(y_unit,equivalencies=u.spectral())
        
        ax.plot(v,self._y,**kwargs)
        
        #Set label of x axis
        ax.set_xlabel(r'$v$' + f' [{v.unit:latex}]')
        
        #Change labels of y axis if it is a normalized spectrum
        if self._normalized:
            ax.set_ylabel(r'$\hat{n}$')
        else:
            ax.set_ylabel(f'Flux [{y.unit:latex}]')
            
        if interval is not None:
            ax.set_xlim(interval[0],interval[1])
        
        return fig
        
    
    def plot(self,x_unit:u.Unit=None, y_unit:u.Unit=None, interval:ArrayLike=None, ax=None,fig_width=10,fig_height=4,**kwargs):
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
        :param fig_width: width of figure, defaults to 10
        :type fig_width: int, optional
        :param fig_height: height of figure, defaults to 4
        :type fig_height: int, optional
        :return: Figure
        """
        
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
            
            fig, ax = plt.subplots(figsize=(fig_width,fig_height))

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
        """Plot zooming into specified intervaks

        :param intervals: intervals that are zoomed into
        :type intervals: list of intervals (interval: list of length 2)
        :param fig_width: width of figure, defaults to 10
        :type fig_width: int, optional
        :param fig_height: height of figure, defaults to 4
        :type fig_height: int, optional
        :param x_unit: x unit for plotting - not changed for whole spectrum, defaults to None
        :type x_unit: u.Unit, optional
        :param y_unit: y unit for plotting - not changed for whole spectrum, defaults to None
        :type y_unit: u.Unit, optional
        :return: figure
        """
        #ToDo: Add option for specifying multiple columns
        
        #Check if intervals have the right shape (x,2) or (2,)
        if (len(np.shape(intervals))==2) and (np.shape(intervals)[-1]==2):
            #multiple intervals
            n_int = len(intervals)
            fig, ax = plt.subplots(n_int,1,figsize=(fig_width,fig_height*n_int))
            for i in range(n_int):
                self.plot(x_unit, y_unit, interval=intervals[i],ax=ax[i],**kwargs)
                
        elif (len(np.shape(intervals))==1) and (np.shape(intervals)[-1]==2):
            #only one interval
            n_int = 1
            fig, ax = plt.subplots(n_int,1,figsize=(fig_width,fig_height*n_int))
            #Only one interval
            self.plot(x_unit, y_unit, interval=intervals,ax=ax,**kwargs)
            
        else:
            logger.error(f'Intervals have the wrong shape {np.shape(intervals)}. Instead, they need to have the shape (x,2) or (2,)')
            raise ValueError
        
        return fig
    
    