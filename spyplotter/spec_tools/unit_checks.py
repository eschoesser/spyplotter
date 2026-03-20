from astropy import units as u
from astropy.coordinates import SpectralCoord, SpectralQuantity

from ..utils.logging import setup_log

logger = setup_log(__name__)


def check_velocity_unit(v):
    """Checking if input v has a unit already, if not, assign km/s

    :param v: velocity
    :type v: float or unit
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """

    if isinstance(
        v, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)
    ) and v.unit.is_equivalent(u.km / u.s):
        logger.debug(f"Use given velocity unit: {v.unit}")
        return v

    elif isinstance(v, (float, int)):
        logger.info("No unit for vrad specified. Thus assuming km/s.")
        return v * u.km / u.s
    else:
        logger.error(
            "Not known format for vrad used. Convert to float or astropy classes Quantity, SpectralCoord or SpectralQuantity"
        )
        raise ValueError


def check_x_unit(x):

    if isinstance(
        x, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)
    ) and x.unit.is_equivalent(u.AA, equivalencies=u.spectral()):
        logger.debug(f"Use given x unit: {x.unit}")
        return SpectralCoord(x)

    elif isinstance(x, (float, int)):
        logger.info("No unit for x specified. Thus assuming Angstroem.")
        return SpectralCoord(x * u.AA)
    else:
        logger.error(
            "Not known format for x used. Convert to float or astropy classes Quantity, SpectralCoord or SpectralQuantity"
        )
        raise ValueError


def check_y_unit(y):
    if isinstance(y, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)):
        # if y has already unit, don't change it
        logger.debug(f"Use given y unit: {y.unit}")
        return y

    elif isinstance(y, (float, int)):

        logger.info("As no unit for y was given, a normalized spectrum is assumed")
        return y * u.dimensionless_unscaled
    else:
        # if unit is specified and y does not have unit, take specified unit
        logger.error(
            "Not known format for y used. Convert to float or astropy classes Quantity, SpectralCoord or SpectralQuantity"
        )
        raise ValueError


def check_distance_unit(d):
    if isinstance(
        d, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)
    ) and d.unit.is_equivalent(u.pc):
        logger.debug(f"Use given distance unit: {d.unit}")
        return d.to(u.pc)

    elif isinstance(d, (float, int)):
        logger.info("No unit for distance specified. Thus assuming parsec.")
        return d * u.pc
    else:
        logger.error(
            "Not known format for distance used. Convert to float or astropy classes Quantity, SpectralCoord or SpectralQuantity"
        )
        raise ValueError


def check_T_unit(T):
    if isinstance(
        T, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)
    ) and T.unit.is_equivalent(u.K):
        logger.debug(f"Use given temperature unit: {T.unit}")
        return T.to(u.K)

    elif isinstance(T, (float, int)):
        logger.info("No unit for temperature specified. Thus assuming Kelvin.")
        return T * u.K
    else:
        logger.error(
            "Not known format for temperature used. Convert to float or astropy classes Quantity, SpectralCoord or SpectralQuantity of unit equivalent to K"
        )
        raise ValueError


def check_column_density_unit(n):
    if isinstance(
        n, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)
    ) and n.unit.is_equivalent(1 / (u.cm**2)):
        logger.debug(f"Use given temperature unit: {n.unit}")
        return n.to(1 / (u.cm**2))

    elif isinstance(n, (float, int)):
        logger.info("No unit for temperature specified. Thus assuming cm^2.")
        return n / (u.cm**2)
    else:
        logger.error(
            "Not known format for column density used. Convert to float or astropy classes Quantity, SpectralCoord or SpectralQuantity of unit equivalent to cm^-2"
        )
        raise ValueError


def doppler_shifted_x(x, vrad):
    # Check and set units of vrad
    vrad = check_velocity_unit(vrad)
    x = check_x_unit(x)

    return x.with_radial_velocity_shift(vrad)


def roman_to_int(s):
    # To convert Roman numerals to integers, we can use a dictionary to map the symbols to their values and then iterate through the string to calculate the total value.
    # May be used for ionization states
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    total = 0
    prev = 0

    for char in reversed(s):
        value = values[char]
        if value < prev:
            total -= value
        else:
            total += value
        prev = value

    return total
