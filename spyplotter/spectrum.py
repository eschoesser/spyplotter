from numpy.typing import ArrayLike
from typing import List
import scipy.constants as const
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SpectralCoord, SpectralQuantity
from pathlib import Path
from astropy.io import fits

from .line_identification import LineIdentifier
from .powr import readWRPlotDatasets

from .spec_tools.convolutions import rotational_broaden_chunks
from .spec_tools.plotting_functions import generate_intervals
from .spec_tools.unit_checks import (
    check_velocity_unit,
    check_x_unit,
    check_y_unit,
    doppler_shifted_x,
)

from .utils.logging import setup_log

logger = setup_log(__name__)


class Spectrum(object):
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        yerr: ArrayLike = None,
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
                self._normalized = False

        elif y_unit == None:
            # if y is not an astropy quantity and no unit was specified, assume a normalized spectrum
            logger.info("As no unit for y was given, a normalized spectrum is assumed")
            self._y = y * u.dimensionless_unscaled
            self._normalized = True
        else:
            # if unit is specified and y does not have unit, take specified unit
            self._y = y * y_unit
            self._normalized = False

        if yerr is not None:
            if isinstance(yerr, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)):
                # if x has already unit, don't change it
                self._yerr = yerr

            elif y_unit == None:
                # if y is not an astropy quantity and no unit was specified, assume a normalized spectrum
                self._yerr = yerr * u.dimensionless_unscaled
            else:
                # if unit is specified and y does not have unit, take specified unit
                self._yerr = yerr * y_unit

            if self._yerr.unit != self._y.unit:
                logger.warning(
                    f"yerr [{self._yerr.unit}] and y [{self._y.unit}] do not have same unit"
                )

        else:
            self._yerr = None

        # mask out regions with zero flux to find segments correctly to not interpolate over zero flux
        # important when combining observed spectra
        mask = self._y.value > 1e-30
        self._x = self._x[mask]
        self._y = self._y[mask]
        self._yerr = self._yerr[mask] if self._yerr is not None else None

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

    @property
    def yerr(self):
        """Return xerr of spectrum

        :param unit: unit of returned value, defaults to None
        :type unit: u.Unit, optional
        :return: _description_
        :rtype: _type_
        """
        return self._yerr

    def x_in_unit(self, unit: u.Unit = None):
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

    def y_in_unit(self, unit: u.Unit = None):
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
    def x_min(self):
        return min(self._x)

    @property
    def x_max(self):
        return max(self._x)

    @property
    def x_lim(self):
        # useful for ax.set_xlim()
        return (min(self._x.value), max(self._x.value))

    @property
    def is_equally_spaced(self):
        """Check if spectrum is equally spaced

        :return: True if spectrum is equally spaced, False otherwise
        :rtype: bool
        """
        return np.all(np.diff(self._x.value) == np.diff(self._x.value)[0])

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
        bin_width: float = None,
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

        sp = cls(x=x, y=y, x_unit=xunit, y_unit=yunit, name=name, vrad=vrad)
        if bin_width is not None:
            sp.bin(bin_width=bin_width, overwrite=True)

        return sp

    @classmethod
    def from_file(
        cls,
        filename,
        xunit: u.Unit = None,
        yunit: u.Unit = None,
        name: str = None,
        vrad=0.0 * u.km / u.s,
        bin_width: float = None,
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

        if len(data[0]) == 2:
            yerr = None
        elif len(data[0]) == 3:
            yerr = data[:, 2]
        else:
            logger.error(
                f"Read data has {len(data[0])} columns but should have either two (x, y) or three (x,y,yerr)"
            )

        sp = cls(
            x=data[:, 0],
            y=data[:, 1],
            yerr=yerr,
            x_unit=xunit,
            y_unit=yunit,
            name=name,
            vrad=vrad,
        )

        if bin_width is not None:
            sp.bin(bin_width=bin_width, overwrite=True)

        return sp

    @classmethod
    def from_fits_eso(
        cls,
        filepath,
        xunit: u.Unit = None,
        yunit: u.Unit = None,
        vrad=0.0 * u.km / u.s,
        bin_width: float = None,
    ):
        """Read spectrum from an ESO data product fits file
        Takes into account that fitsf[1].data has multiple data sets

        :param filepath: path of file, if only path is given and not concrete file, try to open formal.plot
                        Otherwise: open specified file
        :type filepath: string or Path (file or directory)
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
            fitsf = fits.open(path, ignore_missing_simple=True)

        else:
            logger.error("Path does not exist")
            raise ValueError

        data = fitsf[1].data
        n_datasets = len(data)
        err_upper = data["ERROR"]
        # err_lower = data['ERROR_LOWER']
        flux = data["FLUX"]
        lamb = data["WAVELENGTH"]

        sp = cls(
            x=lamb[0][flux[0] > 0],
            x_unit=xunit,
            y=flux[0][flux[0] > 0],
            y_unit=yunit,
            yerr=err_upper[0][flux[0] > 0],
            vrad=-vrad,
        )

        if n_datasets > 1:
            logger.info(
                f"There are {n_datasets} data sets in this file. They will be combined."
            )

            for i in range(n_datasets - 1):
                sp2 = cls(
                    x=lamb[i + 1][flux[i + 1] > 0],
                    x_unit=xunit,
                    y=flux[i + 1][flux[i + 1] > 0],
                    y_unit=yunit,
                    yerr=err_upper[i + 1][flux[i + 1] > 0],
                    vrad=-vrad,
                )
                sp = sp + sp2

        if bin_width is not None:
            sp.bin(bin_width=bin_width, overwrite=True)

        return sp

    def to_file(self, filename: str, **kwargs):
        # advanced todo: save also all parameters that were specified in header
        """Save the wavelength and flux to a .dat file with two columns.

        :param filename: Name of the file to save the data.
        :type filename: str
        """
        # Convert to numpy arrays
        x_values = self.x.value
        y_values = self.y.value

        # Combine x and y values into a single 2D or 3D array depending on if error is given
        if self._yerr is None:
            data = np.column_stack((x_values, y_values))
            header = "Wavelength Flux"
        else:
            yerr_values = self.yerr.value
            data = np.column_stack((x_values, y_values, yerr_values))
            header = "Wavelength Flux FluxError"

        # Save to file
        np.savetxt(filename, data, fmt="%.6e", header=header, **kwargs)

    def convolve_rotation(
        self,
        vsini: float,
        epsilon: float = 0.005,
        edge_handling="firstlast",
        overwrite=False,
        new_spectrum=False,
    ):
        """Convolve spectrum with rotation profile

        :param vrot: rotation velocity
        :type vrot: float or astropy classes Quantity, SpectralCoord or SpectralQuantity
        :raises ValueError: if vrot has not one of required formats
        :param epsilon: limb darkening coefficient
        :type epsilon: float
        :param edge_handling: how to handle edges of spectrum, defaults to 'firstlast'
        :type edge_handling: str, optional
        """

        # check and set velocity unit
        vsini = check_velocity_unit(vsini)

        if not self.is_equally_spaced:
            logger.warning(
                "Spectrum is not equally spaced. Interpolating to equally spaced grid."
            )
            # Interpolate to equally spaced grid
            self.interpolate_equally_spaced()

        new_flux = rotational_broaden_chunks(
            self.x.value,
            self.y.value,
            vsini,
            epsilon=epsilon,
            edge_handling=edge_handling,
        )

        if overwrite:
            # Overwrite spectrum
            logger.debug("Overwrite spectrum which is broadened")
            self._y = new_flux * self.y.unit

        elif new_spectrum:
            # Return new object of spectrum
            logger.debug("Return broadened spectrum")
            return Spectrum(
                x=self.x,
                y=new_flux * self.y.unit,
                x_unit=self.x_unit,
                y_unit=self.y_unit,
                name=self.name,
                vrad=self.vrad,
            )
        else:
            return new_flux * self.y.unit

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
                x=self.x,
                y=self.y,
                x_unit=self.x_unit,
                y_unit=self.y_unit,
                name=self.name,
                vrad=new_vrad,
            )

        else:
            return doppler_shifted_x(self.x, vrad)

    def _get_segments(self, diff_factor=25):
        """
        Observed spectra can consist multiple intervals
        Identify segments in the spectrum based on gaps in x-values.


        :param diff_factor: factor of average spacing at which spectrum is divided in segments, defaults to 2
        :type diff_factor: int, optional
        :return: segments of spectrum
        :rtype: _type_
        """
        x = self._x.value.tolist()
        y = self._y.value.tolist()
        yerr = self._yerr.value.tolist() if self._yerr is not None else None

        # Compute spacing of x
        diffs = np.diff(x)
        avg_diff = np.average(diffs)

        # seperate into segments if differences in x larger than diff_factor* average spacing
        segment_edges = np.where(diffs > diff_factor * avg_diff)[0] + 1

        # create x,y,yerr segments according to computed edges
        x_segments = np.split(x, segment_edges)
        y_segments = np.split(y, segment_edges)
        yerr_segments = np.split(yerr, segment_edges) if yerr is not None else None

        # convert arrays to lists of lists
        x_segments = [seg.tolist() for seg in x_segments]
        y_segments = [seg.tolist() for seg in y_segments]
        yerr_segments = (
            [seg.tolist() for seg in yerr_segments]
            if yerr_segments is not None
            else None
        )

        return (x_segments, y_segments, yerr_segments)

    def __add__(self, other):
        """Combine spectra using signal-to-noise-ratio (SNR) weighted mean
        and appending in non-overlapping regions
        yerr is scaled accordingly

        ToDO: Fix so that sp1+sp2 = sp1+sp2

        :param other: Spectrum that should be added
        :type other: Spectrum
        :return: In overlapping regions, SNR weighted mean is used
        :rtype: Spectrum
        """
        if not isinstance(other, Spectrum):
            raise ValueError("Can only add another Spectrum object.")

        if self.x.unit != other.x.unit:
            raise ValueError(
                f"spectrum1.x [{self.x.unit}] and spectrum2.x [{other.x.unit}] do not have same units. Please convert units first"
            )

        if self.y.unit != other.y.unit:
            raise ValueError(
                f"spectrum1.y and spectrum2.y do not have same units. Please convert units first"
            )

        # Compute segemnts of spectra to find gaps where no data was taken
        x_segments_self, y_segments_self, yerr_segments_self = self._get_segments()
        logger.debug(f"{len(x_segments_self)} segments found for self")
        logger.debug(
            f"Segments for self:{[(min(xs),max(xs)) for xs in x_segments_self]}"
        )
        x_segments_other, y_segments_other, yerr_segments_other = other._get_segments()
        logger.debug(f"{len(x_segments_other)} segments found for other")
        logger.debug(
            f"Segments for other:{[(min(xs),max(xs)) for xs in x_segments_other]}"
        )

        # Order and combine segments by min(x) of each segment to then add them in right order
        combined_x_segments = x_segments_self + x_segments_other

        # Sort the combined lists by min(x) in each segment
        sorted_combined_x_segments = sorted(combined_x_segments, key=lambda x: x[0])

        # Compute indices of the sorted order to then order y and yerr in same way
        sorted_indices = [
            combined_x_segments.index(item) for item in sorted_combined_x_segments
        ]

        # Apply same order to y and yerr
        combined_y_segments = y_segments_self + y_segments_other
        sorted_combined_y_segments = [combined_y_segments[i] for i in sorted_indices]

        if self._yerr is not None:
            combined_yerr_segments = yerr_segments_self + yerr_segments_other
            sorted_combined_yerr_segments = [
                combined_yerr_segments[i] for i in sorted_indices
            ]
        else:
            sorted_combined_yerr_segments = None

        # Start with first segment and then append segments to the right (increasing x)
        x_new = sorted_combined_x_segments[0]
        y_new = sorted_combined_y_segments[0]
        yerr_new = (
            sorted_combined_yerr_segments[0]
            if sorted_combined_yerr_segments is not None
            else None
        )

        for i in range(1, len(sorted_combined_x_segments)):
            # values of ith iteration
            x_i = sorted_combined_x_segments[i]
            y_i = sorted_combined_y_segments[i]
            yerr_i = (
                sorted_combined_yerr_segments[i]
                if sorted_combined_yerr_segments is not None
                else None
            )

            # append spectrum with wavelengths larger than original one
            mask_x_1 = np.array(x_i) > max(x_new)
            if not np.all(~mask_x_1):
                x_new = x_new + np.array(x_i)[mask_x_1].tolist()
                y_new = y_new + np.array(y_i)[mask_x_1].tolist()
                yerr_new = (
                    yerr_new + np.array(yerr_i)[mask_x_1].tolist()
                    if yerr_new is not None
                    else None
                )

            # for overlapping wavelength (x) regions, find min and max of intervals,
            # bin to the same x_values,
            # compute SNR, apply SNR weighted mean
            # Find the overlap interval
            mask_x_2 = ~(mask_x_1)

            if not np.all(~mask_x_2):
                # Interval of overlap
                overlap_start, overlap_end = min(np.array(x_i)[mask_x_2]), max(
                    np.array(x_i)[mask_x_2]
                )
                logger.debug("")

                mask_new_overlap = (np.array(x_new) >= overlap_start) & (
                    np.array(x_new) <= overlap_end
                )
                mask_i_overlap = (np.array(x_i) >= overlap_start) & (
                    np.array(x_i) <= overlap_end
                )

                # Determine common x values
                common_x = np.sort(
                    np.unique(
                        np.concatenate(
                            (
                                np.array(x_new)[mask_new_overlap],
                                np.array(x_i)[mask_i_overlap],
                            )
                        )
                    )
                )

                # Interpolate new onto common x values
                interp_y_new = np.interp(common_x, x_new, y_new)
                interp_yerr_new = (
                    np.interp(common_x, x_new, yerr_new)
                    if yerr_new is not None
                    else None
                )

                ## Interpolate i onto common x values
                interp_y_i = np.interp(common_x, x_i, y_i)
                interp_yerr_i = (
                    np.interp(common_x, x_i, yerr_i) if yerr_i is not None else None
                )

                # Compute SNR at each x
                snr_new = (
                    interp_y_new / interp_yerr_new
                    if interp_yerr_new is not None
                    else np.ones_like(interp_y_new)
                )

                snr_i = (
                    interp_y_i / interp_yerr_i
                    if interp_yerr_i is not None
                    else np.ones_like(interp_y_i)
                )

                total_snr = snr_new + snr_i
                weight_new = snr_new / total_snr
                weight_i = snr_i / total_snr

                snr_weighted_y = weight_new * interp_y_new + weight_i * interp_y_i

                snr_weighted_yerr = (
                    np.sqrt(
                        (weight_new * interp_yerr_new) ** 2
                        + (weight_i * interp_yerr_i) ** 2
                    )
                    if interp_yerr_new is not None
                    else None
                )

                # Combine spectra, evaluate at common_x
                mask1 = np.array(x_new) < overlap_start
                mask2 = np.array(x_new) > overlap_end

                x_new = (
                    np.array(x_new)[mask1].tolist()
                    + common_x.tolist()
                    + np.array(x_new)[mask2].tolist()
                )

                y_new = (
                    np.array(y_new)[mask1].tolist()
                    + snr_weighted_y.tolist()
                    + np.array(y_new)[mask2].tolist()
                )

                yerr_new = (
                    (
                        np.array(yerr_new)[mask1].tolist()
                        + snr_weighted_yerr.tolist()
                        + np.array(yerr_new)[mask2].tolist()
                    )
                    if yerr_new is not None
                    else None
                )

            else:
                logger.debug("No overlap, so no averaging")

        return Spectrum(
            x_new * self.x.unit,
            y_new * self.y.unit,
            yerr_new * self.yerr.unit if self.yerr is not None else None,
        )

    def bin(self, bin_width, overwrite=False, new_spectrum=False):
        # Compute the number of bins
        num_bins = int(np.ceil((self._x.value[-1] - self._x.value[0]) / bin_width))

        # Compute bin edges
        bin_edges = np.linspace(self._x.value[0], self._x.value[-1], num_bins + 1)

        # Initialize arrays to store binned values
        binned_wavelengths = []
        binned_flux = []
        if self._yerr is None:
            binned_flux_error = None
        else:
            binned_flux_error = []

        # Iterate over the bins
        for i in range(num_bins):
            # Find indices of wavelengths falling within the current bin
            bin_indices = np.where(
                (self._x.value >= bin_edges[i]) & (self._x.value < bin_edges[i + 1])
            )

            # Compute the average flux and wavelength in the bin
            if bin_indices[0].size > 0:
                binned_wavelengths.append(np.mean(self._x.value[bin_indices]))
                binned_flux.append(np.mean(self._y.value[bin_indices]))
                if self._yerr is not None:
                    binned_flux_error.append(
                        np.sqrt(np.sum(self._yerr.value[bin_indices] ** 2))
                        / len(self._yerr.value[bin_indices])
                    )

        x, y = (
            SpectralCoord(np.array(binned_wavelengths) * self._x.unit),
            np.array(binned_flux) * self._y.unit,
        )

        if self._yerr is not None:
            yerr = np.array(binned_flux_error) * self._yerr.unit
        else:
            yerr = None

        if overwrite:
            self._x = x
            self._y = y
            self._yerr = yerr

        elif new_spectrum:
            return Spectrum(
                x=x,
                y=y,
                yerr=yerr,
                name=self.name,
            )
        else:
            return x, y, yerr

    def interpolate_equally_spaced(self, dx=None, new_spectrum=False):
        """Interpolate the spectrum to have equally spaced x-values
        :param dx: spacing of x values, defaults to None
        :type dx: float, optional
        :param new_spectrum: if True, return a new spectrum object, defaults to False
        :type new_spectrum: bool, optional
        :return: Interpolated spectrum
        :rtype: Spectrum
        """
        if dx is None:
            dx = np.mean(np.diff(self._x.value))
        # Create a new x array with equally spaced values
        x_new = np.arange(self._x.value[0], self._x.value[-1], dx) * self._x.unit
        # Interpolate the y values to the new x values
        y_new = np.interp(x_new.value, self._x.value, self._y.value)
        # Create a new Spectrum object with the interpolated values
        if self._yerr is not None:
            yerr_new = np.interp(x_new.value, self._x.value, self._yerr.value)

        if new_spectrum:
            logger.debug("Return new object with equally spaced x-values")
            new_spectrum = Spectrum(
                x=x_new,
                y=y_new * self._y.unit,
                yerr=yerr_new * self._yerr.unit if self._yerr is not None else None,
                name=self.name,
            )
            return new_spectrum
        else:
            logger.debug("Overwrite spectrum with equally spaced x-values")
            self._x = x_new
            self._y = y_new * self._y.unit
            if self._yerr is not None:
                self._yerr = np.interp(x_new.value, self._x.value, self._yerr.value)

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
        yshift=0,
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
        :param yshift: shift in y direction, assumed same unit as y
        :type y_unit: optional
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

        # Check yshift unit
        if isinstance(yshift, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)):
            if y.unit == y_unit:
                # if y has already unit, don't change it
                logger.debug(f"Use given yshift unit: {y.unit}")
                y_s = yshift
            else:
                logger.info("converting yshift to same unit as y_unit")
                y_s = yshift.to(y_unit, equivalencies=u.spectral())

        elif isinstance(yshift, (float, int)):
            logger.info("As no unit for y was given, a normalized spectrum is assumed")
            y_s = yshift * y.unit
        else:
            # if unit is specified and y does not have unit, take specified unit
            logger.error(
                "Not known format for y used. Convert to float or astropy classes Quantity, SpectralCoord or SpectralQuantity"
            )
            raise ValueError

        ax.plot(x, y + y_s, **kwargs)

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
        yshift=0,
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
        # ToDo maybe: Check if intervals have the right shape (x,2) or (2,)
        if ax is None:
            if isinstance(intervals, int):
                # if it's of type int, divide x in n intervals given by integer
                intervals = np.array(
                    generate_intervals(
                        interval_start=np.min(self.x_in_unit(x_unit).value),
                        interval_end=np.max(self.x_in_unit(x_unit).value),
                        n_int=intervals,
                    )
                )
        else:
            if np.shape(ax) == ():
                # only one axis, no zoom plot but should still not result in error
                intervals = np.array(ax.get_xlim())
            elif intervals is None:
                # use x limits from previous figure
                intervals = [list(ax[i].get_xlim()) for i in range(len(ax))]

                logger.warning(
                    f"Using following intervals from pre-set values in figure given by ax: \n\t{intervals}"
                )

        if ((len(np.shape(intervals)) == 2) and (np.shape(intervals)[-1] == 2)) and len(
            intervals
        ) > 1:
            # multiple intervals
            n_int = len(intervals)
            if ax is None:
                fig, ax = plt.subplots(
                    n_int, 1, figsize=(fig_width, fig_height * n_int)
                )

            for i in range(n_int):
                self.plot(
                    x_unit,
                    y_unit,
                    yshift=yshift,
                    interval=intervals[i],
                    ax=ax[i],
                    **kwargs,
                )
                if i < n_int - 1:
                    # only labels on lowest x axis
                    ax[i].set_xlabel("")

        elif (len(np.shape(intervals)) == 1) and (np.shape(intervals)[-1] == 2):
            # only one interval with form [a,b]
            n_int = 1
            if ax is None:
                fig, ax = plt.subplots(
                    n_int, 1, figsize=(fig_width, fig_height * n_int)
                )
            # Only one interval
            self.plot(
                x_unit, y_unit, interval=intervals, yshift=yshift, ax=ax, **kwargs
            )
        elif np.shape(intervals) == (1, 2):
            # only one interval with form [[a,b]]
            n_int = 1
            if ax is None:
                fig, ax = plt.subplots(
                    n_int, 1, figsize=(fig_width, fig_height * n_int)
                )
            # Only one interval
            self.plot(
                x_unit, y_unit, interval=intervals[0], yshift=yshift, ax=ax, **kwargs
            )

        else:
            logger.error(
                f"Intervals have the wrong shape {np.shape(intervals)}. Instead, they need to have the shape (x,2) or (2,)"
            )
            raise ValueError

        return ax
