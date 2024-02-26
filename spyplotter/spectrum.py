from numpy.typing import ArrayLike
from typing import List
import scipy.constants as const
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SpectralCoord, SpectralQuantity
from pathlib import Path

from .line_identification import LineIdentifier
from .powr import readWRPlotDatasets
from .utils.logging import setup_log
from .spec_tools.plotting_functions import generate_intervals
from .spec_tools.unit_checks import check_velocity_unit, check_x_unit, doppler_shifted_x

logger = setup_log(__name__)


class Spectrum(object):
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        x_unit: u.Unit = None,
        y_unit: u.Unit = None,
        name: str = None,
        vrad=0 * u.km / u.s,
        line_identifier: LineIdentifier = None,
    ):
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
        if isinstance(x, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)):
            # if x has already unit, don't change it
            logger.info(f"Keeping units of x: {x.unit}")
            self._x = SpectralCoord(x)
        elif x_unit == None:
            # if no unit is given, use nm
            logger.info("As no unit for x was specified, Angstrom are assumed")
            self._x = SpectralCoord(x * u.AA)
        else:
            # if unit is specified and x does not have unit, take specified unit
            self._x = SpectralCoord(x * x_unit)

        if isinstance(y, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)):
            # if y has already unit, don't change it
            logger.info(f"Keeping units of y: {y.unit}")
            self._y = y

            # if pre-specified unit is dimensionless, assume a normalized spectrum
            if y.unit == u.dimensionless_unscaled:
                self._normalized = True
            else:
                self.normalized = False

        elif y_unit == None:
            # if y is not an astropy quantity and no unit was specified, assume a normalized spectrum
            logger.info("As no unit for y was given, a normalized spectrum is assumed")
            self._y = y * u.dimensionless_unscaled
            self._normalized = True
        else:
            # if unit is specified and y does not have unit, take specified unit
            self._y = y * y_unit
            self._normalized = False

        v = check_velocity_unit(vrad)
        self._vrad = v
        self._x = self._x.with_radial_velocity_shift(v)
        self._line_identifier = line_identifier

    def __call__(self):
        # ToDo: when called at a specific x point, use spline or interpolation to evaluate flux at given point?
        return self.spectrum.T

    @property
    def spectrum(self) -> ArrayLike:
        """Convert spectrum in a 2D numpy array

        :return: spectrum as numpy array
        :rtype: ArrayLike
        """
        return np.array([self._x, self._y])

    @property
    def x(self):
        """Return x values of spectrum

        :param unit: unit of returned value, defaults to None
        :type unit: u.Unit, optional
        :return: _description_
        :rtype: _type_
        """
        return self._x

    def x(self, unit: u.Unit = None):
        """Return x values of spectrum

        :param unit: unit of returned value, defaults to None
        :type unit: u.Unit, optional
        :return: _description_
        :rtype: _type_
        """
        if unit == None:
            return self._x
        else:
            return self._x.to(unit, equivalencies=u.spectral())

    def y(self, unit: u.Unit = None):
        if unit == None:
            return self._y
        else:
            return self._y.to(unit, equivalencies=u.spectral())

    @property
    def y(self):
        return self._y

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

    @property
    def line_identifier(self):
        if self._line_identifier is None:
            logger.warning("Line Identifier was not defined yet")
        return self._line_identifier

    @line_identifier.setter
    def line_identifier(self, line_identifier: LineIdentifier):
        self._line_identifier = line_identifier

    @classmethod
    def from_powr(
        cls,
        filepath,
        keywords: List[int] = [""],
        dataset: int = 1,
        xunit: u.Unit = None,
        yunit: u.Unit = None,
        name=None,
        vrad=0.0 * u.km / u.s,
    ):
        """Read spectrum from a PoWR output file

        :param filepath: path of file, if only path is given and not concrete file, try to open formal.plot
                        Otherwise: open specified file
        :type filepath: string or Path (file or directory)
        :param keywords: list of keywords that are read after each other, defaults to ['']
        :type keywords: List[int], optional
        :param dataset: number of data set that is read, defaults to 1
        :type dataset: int, optional
        :param xunit: unit of x_values, defaults to None (later Angstrom)
        :type xunit: u.Unit, optional
        :param yunit: unit of y values, defaults to None (later normalized)
        :type yunit: u.Unit, optional
        :param name: name of spectrum, defaults to None
        :type name: string, optional
        :param vrad: radial velocity that should be applied to spectrum, defaults to 0.*u.km/u.s
        :type vrad: float, SpectralCoord, SpectralQuantity or u.quantity.Quantity, optional
        :raises ValueError: if path or formal.plot in directory does not exist
        :return: Spectrum object
        :rtype: Spectrum
        """

        path = Path(filepath)

        # Check if model path exists
        if path.exists():
            if path.is_file():
                x, y = readWRPlotDatasets(filepath, keywords, dataset)

            elif (path / "formal.plot").exists():
                logger.info("Reading formal.plot")
                x, y = readWRPlotDatasets(path / "formal.plot", keywords, dataset)
        else:
            logger.error("Path does not exist")
            raise ValueError

        if yunit is None:
            # search for signs of a flux calibrated spectrum, otherwise assume normalized spectrum
            if (
                ("CONTINUUM" in keywords[0]) or ("EMERGENT" in keywords[0])
            ) or dataset > 1:
                yunit = u.erg / u.s / u.cm**2 / u.AA
                logger.info(
                    f"Flux calibrated spectrum at 10 pc. Thus using {yunit} as y unit."
                )
            else:
                logger.info(
                    "No flux unit specified and no signs for y units detected. Thus assuming normalized spectum."
                )
                yunit = None

        return cls(x, y, xunit, yunit, name, vrad)

    @classmethod
    def from_file(
        cls,
        filename,
        xunit: u.Unit = None,
        yunit: u.Unit = None,
        name: str = None,
        vrad=0.0 * u.km / u.s,
        **kwargs,
    ):
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
            data = np.loadtxt(filename, **kwargs)
        else:
            logger.error("Path does not exist")
            raise ValueError

        return cls(data[:, 0], data[:, 1], xunit, yunit, name, vrad)

    @classmethod
    def from_cmfgen(cls, filepath, name: str = None):
        # todo: write a function that imports from a CMFGEN output file a specific simulated Spectrum class
        pass

    def to_file(self, filename: str):
        # todo: write a function that saves table of current spectrum in a csv file
        # advanced todo: save also all parameters that were specified in header
        pass

    def convert_units(self, x_unit: u.Unit = None, y_unit: u.Unit = None):
        """Converts units and overwrites them in spectrum

        :param x_unit: unit for x_values, defaults to None
        :type x_unit: u.Unit, optional
        :param y_unit: unit for y_values, defaults to None
        :type y_unit: u.Unit, optional
        """

        if (x_unit is None) and (y_unit is None):
            logger.info(
                "Specify which (x_unit or y_unit) should be converted and specify the desired unit using astropy.units"
            )
            pass

        if x_unit is not None:
            logger.debug(f"The unit of all x values is changed to {x_unit}")
            self._x = self._x.to(x_unit, equivalencies=u.spectral())

        if y_unit is not None:
            logger.debug(f"The unit of all y values is changed to {y_unit}")
            self._x = self._y.to(y_unit, equivalencies=u.spectral())

    def apply_shift_vrad(self, vrad, overwrite=False, new_spectrum=False):
        """Apply radial shift to spectrum, choose if spectrum is overwritten or new spectrum is returned

        :param vrad: radial velocity
        :type vrad: float or astropy classes Quantity, SpectralCoord or SpectralQuantity
        :raises ValueError: if vrad has not one of required formats
        """
        # warn if spectrum was shifted before
        if self.vrad != 0:
            logger.warning(
                f"Spectrum was already shifted using vrad={self.vrad}. \nNow total shift of spectrum: vradtot = {self.vrad + vrad}"
            )

        if overwrite:
            # Overwrite spectrum
            logger.debug("Overwrite spectrum when applying velocity shift")
            self._vrad = vrad + self.vrad
            self._x = doppler_shifted_x(self.x, vrad)

            return self._x

        elif new_spectrum:
            # Return new object of spectrum
            logger.debug("Return new object with shifted x-values")

            # check and set velocity unit
            v = check_velocity_unit(vrad)

            new_vrad = self._vrad + v

            return Spectrum(
                self.x, self.y, self.x_unit, self.y_unit, self.name, new_vrad
            )

        else:
            return doppler_shifted_x(self.x, vrad)

    def to_velocity_space(
        self,
        x_rest,
        doppler_convention="optical",
        v_unit=u.km / u.s,
        vrad=0.0 * u.km / u.s,
    ):
        """Convert the x-values to velocity space, apply radial shift if specified

        :param x_rest: Doppler rest value used for conversion of spectrum to velocity space
        :type x_rest: float or astropyQuantity,SpectralCoord,SpectralQuantity
        :param doppler_convention: Doppler convention, defaults to 'optical'
        :type doppler_convention: str, optional
        :param v_unit: velocity unit, defaults to u.km/u.s
        :type v_unit: u.Unit, optional
        :param vrad: radial velocity, defaults to 0.*u.km/u.s
        :type vrad: float or astropyQuantity,SpectralCoord,SpectralQuantity, optional
        :return: velocity values corresponding to spectrum and x_rest
        :rtype: ArrayLike
        """

        if vrad != 0:
            vrad = check_velocity_unit(vrad)
            x = self.apply_shift_vrad(vrad, overwrite=False, new_spectrum=False)
        else:
            x = self._x

        x_rest = check_x_unit(x_rest)

        return x.to(
            v_unit,
            doppler_convention=doppler_convention,
            doppler_rest=x_rest,
        )

    def plot_velocity(
        self,
        x_rest,
        doppler_convention="optical",
        v_unit=u.km / u.s,
        vrad=0.0 * u.km / u.s,
        ax=None,
        fig_width=10,
        fig_height=4,
        y_unit: u.Unit = None,
        interval: ArrayLike = None,
        zero_vline=True,
        **kwargs,
    ):
        """Convert the x-values of the spectrum to velocity space and plot the spectrum

        :param x_rest: rest value, used to convert to velocity space
        :type x_rest: float or astropyQuantity,SpectralCoord,SpectralQuantity
        :param doppler_convention: Doppler convention, defaults to 'optical'
        :type doppler_convention: str, optional
        :param v_unit: velocity unit for plotting on x axis, defaults to u.km/u.s
        :type v_unit: u.Unit, optional
        :param vrad: radial velocity, defaults to 0.*u.km/u.s
        :type vrad: float or astropyQuantity,SpectralCoord,SpectralQuantity, optional
        :param ax: Axis of plot, defaults to None
        :type ax: matplotlib axis, optional
        :param fig_width: Figure width, defaults to 10
        :type fig_width: int, optional
        :param fig_height: Figure height, defaults to 4
        :type fig_height: int, optional
        :param y_unit: unit of y axis, defaults to None
        :type y_unit: u.Unit, optional
        :param interval: Interval for zoom on x-axis
                        y-axis is zoomed accordingly (98%*y_min,102%*y_min), defaults to None
        :type interval: ArrayLike, optional
        :param zero_vline: decides if vertical line at zero is plotted, defaults to True
        :type zero_vline: bool, optional
        :raises ValueError: if interval has wrong type
        :return: axis of plot
        :rtype: matplotlib axis
        """
        # TODO: add idents

        v = self.to_velocity_space(
            x_rest=x_rest,
            doppler_convention=doppler_convention,
            v_unit=v_unit,
            vrad=vrad,
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        else:
            fig = ax.get_figure()

        # changing y unit if specified
        if y_unit is None:
            logger.debug(
                f"Using the following pre-specified units:\n\ty:{self._y.unit}"
            )
            y = self._y
        else:
            logger.debug(f"Using following new units:\n\ty: {y_unit}")
            y = self._y.to(y_unit, equivalencies=u.spectral())

        # zoom into set x-region
        if interval is not None:
            # Check if interval has length 2
            if len(interval) == 2:
                # check if interval has items
                if all(isinstance(item, (float, int)) for item in interval):
                    logger.debug(f"Use same units as v_unit [{v_unit}]")
                    x_min = interval[0] * v_unit
                    x_max = interval[1] * v_unit
                    ax.set_xlim(x_min.value, x_max.value)
                    # Find indices within the specified x-interval
                    indices = np.where((v >= x_min) & (v <= x_max))
                    min_y = np.min(y[indices])
                    max_y = np.max(y[indices])
                    # zoom into corresponding v values
                    ax.set_ylim(0.98 * min_y.value, 1.02 * max_y.value)
                elif all(
                    isinstance(
                        item, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)
                    )
                    for item in interval
                ):
                    if all(item.unit.is_equivalent(u.m / u.s) for item in interval):
                        # if interval has also velocity units
                        v_min = interval[0]
                        v_max = interval[1]
                        ax.set_xlim(v_min.value, v_max.value)
                        # Find indices within the specified x-interval
                        indices = np.where((v >= v_min) & (v <= v_max))
                        min_y = np.min(y[indices])
                        max_y = np.max(y[indices])
                        # Zoom into corresponding v values
                        ax.set_ylim(0.98 * min_y.value, 1.02 * max_y.value)

                    elif all(
                        item.unit.is_equivalent(self.x_unit, equivalencies=u.spectral())
                        for item in interval
                    ):
                        # if interval items have unit of wavelength, frequency or energy, convert them to vel space
                        x_min = check_x_unit(interval[0])
                        x_max = check_x_unit(interval[1])
                        # convert to velocity space
                        x_rest = check_x_unit(x_rest)
                        v_min = x_min.to(
                            v_unit,
                            doppler_convention=doppler_convention,
                            doppler_rest=x_rest,
                        )
                        v_max = x_max.to(
                            v_unit,
                            doppler_convention=doppler_convention,
                            doppler_rest=x_rest,
                        )
                        # set x limit
                        ax.set_xlim(v_min.value, v_max.value)
                        # Find indices within the specified x-interval
                        indices = np.where((v >= v_min) & (v <= v_max))
                        min_y = np.min(y[indices])
                        max_y = np.max(y[indices])
                        # Zoom into corresponding v values
                        ax.set_ylim(0.98 * min_y.value, 1.02 * max_y.value)

                    else:
                        logger.error(
                            f"Use velocity, wavelnegth, energy or frquency type units when specifying the interval"
                        )

                else:
                    logger.error(
                        f"Interval has wrong type. Use ArrayLike of length 2 and float, int or a Quantity using astropy"
                    )
                    raise ValueError
            else:
                logger.error(f"Interval has a length of {interval} which is unequal 2")
                raise ValueError

        ax.plot(v, self._y, **kwargs)

        # Set label of x axis
        ax.set_xlabel(r"$v$" + f" [{v.unit:latex}]")

        # Change labels of y axis if it is a normalized spectrum
        if self._normalized:
            ax.set_ylabel(r"$\hat{n}$")
        else:
            ax.set_ylabel(f"Flux [{y.unit:latex}]")

        if zero_vline:
            ax.axvline(0, color="grey", ls="--")

        return ax

    def plot(
        self,
        x_unit: u.Unit = None,
        y_unit: u.Unit = None,
        interval: ArrayLike = None,
        ax=None,
        fig_width=10,
        fig_height=4,
        **kwargs,
    ):
        """Function to plot the spectrum
        The units can be changed for the plotting. However,
        the units are only changed for the plot and not the whole class

        :param x_unit: unit of x-values, defaults to None
        :type x_unit: u.Unit, optional
        :param y_unit: unit of y-values, defaults to None
        :type y_unit: u.Unit, optional
        :param interval: interval on x-axis in which it is zoomed in,
                        y-axis is zoomed accordingly (98%*y_min,102%*y_min)
        :type interval: ArrayLike, optional
        :param ax: can be specified to plot in a previously created figure, defaults to None
        :type ax: Axis, optional
        :param fig_width: width of figure, defaults to 10
        :type fig_width: int, optional
        :param fig_height: height of figure, defaults to 4
        :type fig_height: int, optional
        :raises ValueError: if interval has wrong type or shape
        :return: axis
        """

        # TODO: add idents!!

        # does not convert whole spectrum - only for plotting
        if x_unit is None:
            logger.debug(
                f"Using the following pre-specified units:\n\tx:{self._x.unit}"
            )
            x = self._x
        else:
            logger.debug(f"Using following new units:\n\tx: {x_unit}")
            x = self._x.to(x_unit, equivalencies=u.spectral())

        if y_unit is None:
            logger.debug(
                f"Using the following pre-specified units:\n\ty:{self._y.unit}"
            )
            y = self._y
        else:
            logger.debug(f"Using following new units:\n\ty: {y_unit}")
            y = self._y.to(y_unit, equivalencies=u.spectral())

        if ax is None:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        else:
            fig = ax.get_figure()

        ax.plot(x, y, **kwargs)

        # Change labels of x axis dependent on unit type
        if x.unit.physical_type == u.m.physical_type:
            logger.debug("wavelength like units used")
            # wavelength type unit
            ax.set_xlabel(r"$\lambda$" + f" [{x.unit:latex}]")
        elif x.unit.physical_type == u.Hz.physical_type:
            logger.debug("frequency like units used")
            # frequency type unit
            ax.set_xlabel(r"$\nu$" + f" [{x.unit:latex}]")
        elif x.unit.physical_type == u.J.physical_type:
            logger.debug("energy like units used")
            # energy type unit
            ax.set_xlabel(r"$E$" + f" [{x.unit:latex}]")

        # Change labels of y axis if it is a normalized spectrum
        if self._normalized:
            ax.set_ylabel(r"$\hat{n}$")
        else:
            ax.set_ylabel(f" Flux [{y.unit:latex}]")

        # zoom into set x-region
        if interval is not None:
            # Check if interval has length 2
            if len(interval) == 2:
                # check if interval has items
                if all(isinstance(item, (float, int)) for item in interval):
                    logger.debug(f"Use same units as x [{x.unit}]")
                    x_min = interval[0] * x.unit
                    x_max = interval[1] * x.unit
                    ax.set_xlim(x_min.value, x_max.value)
                    # Find indices within the specified x-interval
                    indices = np.where((x >= x_min) & (x <= x_max))
                    # Adapt only ylim if spectrum lies within xlim
                    if len(y[indices]) > 0:
                        min_y = np.min(y[indices])
                        max_y = np.max(y[indices])
                        # zoom into corresponding v values
                        ax.set_ylim(0.98 * min_y.value, 1.02 * max_y.value)
                elif all(
                    isinstance(
                        item, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)
                    )
                    for item in interval
                ):
                    # if interval items have unit of wavelength, frequency or energy, convert them to vel space
                    x_min = check_x_unit(interval[0])
                    x_max = check_x_unit(interval[1])
                    # set x limit
                    ax.set_xlim(x_min.value, x_max.value)
                    # Find indices within the specified x-interval
                    indices = np.where((x >= x_min) & (x <= x_max))
                    # Adapt only ylim if spectrum lies within xlim
                    if len(y[indices]) > 0:
                        min_y = np.min(y[indices])
                        max_y = np.max(y[indices])
                        # Zoom into corresponding v values
                        ax.set_ylim(0.98 * min_y.value, 1.02 * max_y.value)

                else:
                    logger.error(
                        f"Use wavelength,frquency or energy type units when specifying the interval"
                    )

            else:
                logger.error(f"Interval has a length of {interval} which is unequal 2")
                raise ValueError

        return ax

    def plot_zoom(
        self,
        intervals: ArrayLike,
        ax: plt.Axes = None,
        fig_width=10,
        fig_height=4,
        x_unit: u.Unit = None,
        y_unit: u.Unit = None,
        **kwargs,
    ):
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

        # Check if intervals have the right shape (x,2) or (2,)
        if isinstance(intervals, int):
            # if it's of type int, divide x in n intervals given by integer
            intervals = np.array(
                generate_intervals(
                    interval_start=np.min(self.x(x_unit).value),
                    interval_end=np.max(self.x(x_unit).value),
                    n_int=intervals,
                )
            )

        if (len(np.shape(intervals)) == 2) and (np.shape(intervals)[-1] == 2):
            # multiple intervals
            n_int = len(intervals)
            if ax is None:
                fig, ax = plt.subplots(
                    n_int, 1, figsize=(fig_width, fig_height * n_int)
                )

            for i in range(n_int):
                self.plot(x_unit, y_unit, interval=intervals[i], ax=ax[i], **kwargs)

        elif (len(np.shape(intervals)) == 1) and (np.shape(intervals)[-1] == 2):
            # only one interval
            n_int = 1
            if ax is None:
                fig, ax = plt.subplots(
                    n_int, 1, figsize=(fig_width, fig_height * n_int)
                )
            # Only one interval
            self.plot(x_unit, y_unit, interval=intervals, ax=ax, **kwargs)

        else:
            logger.error(
                f"Intervals have the wrong shape {np.shape(intervals)}. Instead, they need to have the shape (x,2) or (2,)"
            )
            raise ValueError

        return ax
