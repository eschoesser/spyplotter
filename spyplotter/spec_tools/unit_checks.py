from astropy import units as u
from astropy.coordinates import SpectralCoord, SpectralQuantity

from ..utils.logging import setup_log
logger = setup_log(__name__)

def check_velocity_unit(v):
        
        if isinstance(v,(u.quantity.Quantity,SpectralCoord,SpectralQuantity)) and v.unit.is_equivalent(u.km/u.s):
            logger.debug(f'Use given velocity unit: {v.unit}')
            return v
            
        elif isinstance(v,(float,int)):
            logger.info('No unit for vrad specified. Thus assuming km/s.')
            return v * u.km / u.s
        else:
            logger.error('Not known format for vrad used. Convert to float or astropy classes Quantity, SpectralCoord or SpectralQuantity')
            raise ValueError
        
def check_x_unit(x):
    
    if isinstance(x,(u.quantity.Quantity,SpectralCoord,SpectralQuantity)) and x.unit.is_equivalent(u.AA,equivalencies=u.spectral()):
        logger.debug(f'Use given x unit: {x.unit}')
        return SpectralCoord(x)
        
    elif isinstance(x,(float,int)):
        logger.info('No unit for x specified. Thus assuming Angstroem.')
        return SpectralCoord(x * u.AA)
    else:
        logger.error('Not known format for x used. Convert to float or astropy classes Quantity, SpectralCoord or SpectralQuantity')
        raise ValueError
    
def doppler_shifted_x(x,vrad):
    #Check and set units of vrad
    vrad = check_velocity_unit(vrad)
    x = check_x_unit(x)
    
    return x.with_radial_velocity_shift(vrad)