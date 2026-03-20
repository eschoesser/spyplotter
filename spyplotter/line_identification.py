import itertools
import numpy as np
import re
import yaml
import matplotlib.pyplot as plt
from itertools import chain
from astropy import units as u
from astropy.coordinates import SpectralCoord, SpectralQuantity

from .powr import wrplot_to_tex
from .utils.logging import setup_log

logger = setup_log(__name__)

from .spec_tools.unit_checks import check_velocity_unit, doppler_shifted_x


class SpectralLine:
    def __init__(
        self,
        ion_name,
        wavelengths,
        plotting_style_dict={},
        x_unit=u.AA,
        vrad=0 * u.km / u.s,
    ):
        self.ion_name = ion_name

        self.x_unit = x_unit
        v = check_velocity_unit(vrad)
        self._vrad = v

        # Check how nested wavelength array is
        # Make sure self._wavelengths is list of lists
        if isinstance(wavelengths, (int, float)):
            self._wavelengths = [
                [SpectralCoord(wavelengths, unit=x_unit).with_radial_velocity_shift(v)]
            ]
        elif isinstance(wavelengths[0], (int, float)):
            self._wavelengths = [
                [
                    SpectralCoord(wavelength, unit=x_unit).with_radial_velocity_shift(v)
                    for wavelength in wavelengths
                ]
            ]
        elif isinstance(wavelengths[0], (list, tuple, np.ndarray)):
            self._wavelengths = [
                [
                    SpectralCoord(wavelength, unit=x_unit).with_radial_velocity_shift(v)
                    for wavelength in sublist
                ]
                for sublist in wavelengths
            ]
        else:
            logger.error(
                "wavelengths have wrong type. Make sure that it is a list/ tuple/ array like of floats."
            )

        self._plotting_style = plotting_style_dict

    @property
    def wavelengths(self):
        return self._wavelengths

    @property
    def wavelengths_vals(self):
        return [
            [float(wavelength.value) for wavelength in sublist]
            for sublist in self._wavelengths
        ]

    @property
    def plotting_style(self):
        return self._plotting_style

    @property
    def vrad(self):
        return self._vrad

    def __add__(self, other):
        if not isinstance(other, SpectralLine):
            raise ValueError("Can only add SpectralLine objects.")

        # Check if the ion names match
        if self.ion_name != other.ion_name:
            raise ValueError("Ion names must match for merging SpectralLines.")

        if self.x_unit != other.x_unit:
            other.convert_unit_to(self.x_unit)

        # Combine wavelengths
        combined_wavelengths = self._wavelengths + other.wavelengths

        # Combine plotting styles
        combined_plotting_style = {**self.plotting_style, **other.plotting_style}

        # Create a new SpectralLine instance with the merged data
        merged_spectral_line = SpectralLine(
            ion_name=self.ion_name,
            wavelengths=combined_wavelengths,
            plotting_style_dict=combined_plotting_style,
            x_unit=self.x_unit,
        )

        return merged_spectral_line

    def apply_shift_vrad(self, vrad, overwrite=False, new_spectral_line=False):
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
            self._wavelengths = [
                [doppler_shifted_x(wavelength, vrad) for wavelength in sublist]
                for sublist in self._wavelengths
            ]

            return self._wavelengths

        elif new_spectral_line:
            # Return new object of spectrum
            logger.debug("Return new object with shifted x-values")

            # check and set velocity unit
            v = check_velocity_unit(vrad)

            new_vrad = self._vrad + v
            return SpectralLine(
                self.ion_name, self.wavelengths_vals, self.x_unit, new_vrad
            )

        else:
            return [
                [doppler_shifted_x(wavelength, vrad) for wavelength in sublist]
                for sublist in self._wavelengths
            ]

    def convert_unit_to(self, new_unit):
        self._wavelengths = [
            [wavelength.to(new_unit) for wavelength in sublist]
            for sublist in self._wavelengths
        ]

    def __str__(self):
        return f"SpectralLine({self.ion_name}:\n\t{self.wavelengths_vals},\n\t{self.plotting_style})"

    def to_dict(self):
        return {
            self.ion_name: {
                "wavelengths": self.wavelengths_vals,
                "plotting_style": self.plotting_style,
            }
        }


class LineIdentifier:
    # TODO: add units, possibility for unit adaptions,make sure it works even if entries of dict_lines have different units
    def __init__(self, spectral_lines={}, x_unit=None, vrad=0 * u.km / u.s):
        """_summary_

        :param spectral_lines: dictionary, defaults to {}
        :type spectral_lines: dict, optional
        :param x_unit: _description_, defaults to None
        :type x_unit: _type_, optional
        :param vrad: _description_, defaults to 0*u.km/u.s
        :type vrad: _type_, optional
        """
        # have_all_units = all(isinstance(value, (u.quantity.Quantity,SpectralCoord,SpectralQuantity)) for value in dict_lines.values())

        if x_unit == None:
            # if no unit is given, use nm
            logger.info("As no x_unit was specified, Angstrom are assumed")
            self._x_unit = u.AA

        self._spectral_lines = spectral_lines
        self._x_unit = x_unit

        v = check_velocity_unit(vrad)
        self._vrad = v
        if self._vrad.value != 0:
            self.apply_shift_vrad(self._vrad)

    def __str__(self):
        str_dict = self.to_dict()
        result = ""
        for i, line in str_dict.items():
            result += f"{i}: {str(line)}\n"
        return result

    @property
    def spectral_lines(self):
        # dictionary of SpectralLine
        return self._spectral_lines

    @property
    def x_unit(self):
        # dictionary of SpectralLine
        return self._x_unit

    @property
    def ions(self):
        return list(self._spectral_lines.keys())

    @property
    def wavelengths(self):
        # nested list of all wavelengths for all ions
        return [
            self._spectral_lines[ion_name].wavelengths_vals
            for ion_name in self._spectral_lines.keys()
        ]

    @property
    def wavelengths_flattened(self):
        # flattened list of all wavelengths for all ions
        return [
            item
            for sublist in self.wavelengths
            for subsublist in sublist
            for item in subsublist
        ]

    @property
    def vrad(self):
        return self._vrad

    def update_plotting_style_ion(self, ion_name, new_plotting_style):
        if ion_name in self._spectral_lines:
            self._spectral_lines[ion_name].plotting_style.update(new_plotting_style)

    def update_plotting_style_all(self, new_plotting_style):
        for ion_name in self.ions:
            if ion_name in self._spectral_lines:
                self._spectral_lines[ion_name].plotting_style.update(new_plotting_style)

    def get_ion_lines(self, ion_name):
        # wavelengths of ion lines
        if ion_name in self._spectral_lines:
            return self._spectral_lines[ion_name].wavelengths_vals
        else:
            logger.error("There are no lines for chosen ion")

    def add_spectral_line(self, spectral_line, x_unit=u.AA):
        """Add Spectral Line to Line Identification
        If ion already exists, plotting style is updated to newly given type

        :param spectral_line: contains information about added lines
        :type spectral_line: Spectral Line
        """
        ion_name = spectral_line.ion_name

        if ion_name in self._spectral_lines:
            self._spectral_lines[ion_name] = (
                self._spectral_lines[ion_name] + spectral_line
            )
        else:
            self._spectral_lines.update({ion_name: spectral_line})

    def convert_units(self, new_unit):
        for ion_name, spectral_line in self._spectral_lines.items():
            spectral_line.convert_unit_to(new_unit)

        self._x_unit = new_unit

    def apply_shift_vrad(self, vrad):
        """Apply radial shift to spectrum, choose if spectrum is overwritten or new spectrum is returned

        :param vrad: radial velocity
        :type vrad: float or astropy classes Quantity, SpectralCoord or SpectralQuantity
        :raises ValueError: if vrad has not one of required formats
        """
        # warn if spectrum was shifted before
        if self.vrad != 0:
            logger.warning(
                f"LineIdentifier was already shifted using vrad={self.vrad}. \nNow total shift of spectrum: vradtot = {self.vrad + vrad}"
            )

        # Overwrite spectrum
        logger.debug("Overwrite LineIdentifier when applying velocity shift")
        self._vrad = vrad + self.vrad
        for ion_name, spectral_line in self._spectral_lines.items():
            spectral_line.apply_shift_vrad(vrad, overwrite=True)

        return self.wavelengths

    @classmethod
    def from_yaml(cls, file_path, x_unit=u.AA, vrad=0 * u.km / u.s):
        # Read Line Identification class from yaml file
        with open(file_path, "r") as yaml_file:
            spectral_lines_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        return cls.from_dict(spectral_lines_dict, x_unit=x_unit, vrad=vrad)

    def to_yaml(self, file_path):
        # Write dictionary to yaml file
        with open(file_path, "w") as yaml_file:
            yaml.dump(self.to_dict(), yaml_file, default_flow_style=False)

    @classmethod
    def from_dict(cls, spectral_lines_dict, x_unit=u.AA, vrad=0 * u.km / u.s):
        # Create dictionary of SpectralLine type from dictionary of dictionaries
        spectral_lines = {}
        for ion, line in spectral_lines_dict.items():
            if isinstance(line, dict):
                if "plotting_style" in line:
                    spectral_lines.update(
                        {
                            ion: SpectralLine(
                                ion_name=ion,
                                wavelengths=line["wavelengths"],
                                plotting_style_dict=line["plotting_style"],
                                x_unit=x_unit,
                            )
                        }
                    )
                else:
                    spectral_lines.update(
                        {
                            ion: SpectralLine(
                                ion_name=ion,
                                wavelengths=line["wavelengths"],
                                x_unit=x_unit,
                            )
                        }
                    )
            elif isinstance(line, (float, int, list, tuple, np.ndarray)):
                spectral_lines.update(
                    {ion: SpectralLine(ion_name=ion, wavelengths=line, x_unit=x_unit)}
                )

        return cls(spectral_lines, x_unit, vrad)

    def to_dict(self):
        # Convert all SpectralLine objects to dictionary
        # Yields nested dictionary
        spectral_lines_dict = {}
        for line in self._spectral_lines.values():
            spectral_lines_dict.update(line.to_dict())
        return spectral_lines_dict

    @classmethod
    def from_powr_identfile(cls, filename, keyword="", x_unit: u.Unit = u.AA):
        """Reads input ident file of WRPlot and converts it to
        a dictionary containing the information of the lines
        and a dictionary containing the text style properties

        :param filepath: file path
        :type filepath: string or Path
        :param keyword: keyword in ident file that is looked for to find
                        corresponding set of lines
        :type keyword: string
        :return: dictionary containing string in wrplot format and line positions
        :rtype: _type_
        """
        endkeys = ["FINISH", "END"]
        spectral_lines = {}
        with open(filename, "r") as file:
            foundkey = keyword == ""
            for line in file:
                if not foundkey:
                    keypos = line.find(keyword)
                    if keypos == -1:
                        # skip iteration if keyword was not found yet
                        continue
                    else:
                        foundkey = True

                rawline = line.rstrip()
                # Define the known keywords at beginning of lines that are read
                known_keywords = [r"\IDENT", "\IDMULT"]
                for known_keyword in known_keywords:
                    # Use regular expression to match the known keyword, floats, and the rest of the string
                    pattern = r"({})\s+([\d.]+(?:\s+[\d.]+)*)\s*(.*)".format(
                        re.escape(known_keyword)
                    )

                    # Use re.match to find the pattern in the read line
                    match = re.match(pattern, rawline)
                    if match is not None:
                        # group string into beginning key word, floats and text_label
                        _, floats, text_label = match.groups()
                        floats_list = [
                            float(num) for num in re.findall(r"[\d.]+", floats)
                        ]
                        # convert dictionary keys to latex format and plotting dictionary
                        ion_name, plotting_dict = wrplot_to_tex(text_label.strip())

                        if ion_name in spectral_lines:
                            spectral_lines[ion_name] = spectral_lines[
                                ion_name
                            ] + SpectralLine(ion_name, floats_list, plotting_dict)
                        else:
                            sl = SpectralLine(
                                ion_name=ion_name,
                                wavelengths=[floats_list],
                                plotting_style_dict=plotting_dict,
                                x_unit=x_unit,
                            )
                            spectral_lines.update({ion_name: sl})
                        break
                # if one of end keys is read, stop reading
                if any(rawline.strip().startswith(endkey) for endkey in endkeys):
                    break

        return cls(spectral_lines, x_unit)

    def plot(
        self,
        base_yoff=0.9,
        root=0.05,
        stem=0.05,
        stem_xoff_rel_cen=0,
        text_yoff=0.03,
        ax=None,
        line_kwargs={"linewidth": 0.7, "color": "k"},
        text_kwargs={
            "fontsize": 10,
            "rotation": 90,
            "color": "k",
            "ha": "center",
            "va": "bottom",
        },
        default_kwargs=True,
    ):
        """

                            NAME
                                (text_yoff)
        (stem)  _____C________|    (stem_xoff_rel_cen)
        (root) |          |
                                (base_yoff)
        vvvvv\ /vvvvvvv\    /vvvvvvvvvvvvvvv-spectrum-vvvv
            V         |  |
                        \/
        the text height and vertical offsets (base_yoff,root,stem, text_yoff)
        are scaled relative to the y-axis range.

        stem_xoff_rel_cen : stem x offset relative to the center (C, average wavelength of the lambdas (lambN set))
        base_yoff         : base of the ident y offset
        root              : the length of the "root", the line which points to the spectral line
        stem              : the length of the "stem", the line which points to the label NAME (spectral line id)
        text_yoff         : text y offset relative to the top of the stem
        line_kwargs       : dictionary for customizing vlines and hlines
        default_kwargs    : if set True and new kwargs for text_kwargs and line_kwargs are chosen, the deafault values used but are updated

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            xlim = None
        else:
            fig = ax.get_figure()
            xlim = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

        y_range = ymax - ymin
        # Scale offsets to y-range
        base_yoff_scaled = ymin + base_yoff * y_range
        root_scaled = root * y_range
        stem_scaled = stem * y_range
        text_yoff_scaled = text_yoff * y_range

        if default_kwargs:
            # use default plotting style and only update explicitly changed values
            line_kwargs_default = {"linewidth": 0.7, "color": "k"}
            text_kwargs_default = {
                "fontsize": 10,
                "rotation": 90,
                "color": "k",
                "ha": "center",
                "va": "bottom",
            }
            # Update line style
            line_kwargs_default.update(line_kwargs)
            line_kwargs = line_kwargs_default.copy()
            # Update text style
            text_kwargs_default.update(text_kwargs)
            text_kwargs = text_kwargs_default.copy()

        # flattened list of all wavelengths
        wavel = self.wavelengths_flattened
        # y value for vertical root lines
        ymin = base_yoff_scaled
        ymax = base_yoff_scaled + root_scaled
        # vertical root lines which point to spectral lines
        ax.vlines(wavel, ymin=ymin, ymax=ymax, **line_kwargs)

        # list of minimum and maximum x values if there are multiplets
        xmin_xmax_values = np.array(
            [[min(value), max(value)] for ion in self.wavelengths for value in ion]
        )

        # mask for xmin and xmax of multiplet lines
        mask = xmin_xmax_values[:, 0] != xmin_xmax_values[:, 1]
        xmin_xmax_multiplet_values = xmin_xmax_values[mask]
        # all horizontal lines are on same y value
        y = [base_yoff_scaled + root_scaled] * len(xmin_xmax_multiplet_values)
        # horizontal lines connecting root lines corresponding to one multiplet
        ax.hlines(
            y,
            xmin=xmin_xmax_multiplet_values[:, 0],
            xmax=xmin_xmax_multiplet_values[:, 1],
            **line_kwargs,
        )

        # stem line x value to label
        stem_lamb = np.mean(xmin_xmax_values, axis=1) + stem_xoff_rel_cen
        # constant y values
        ymin = base_yoff_scaled + root_scaled
        ymax = base_yoff_scaled + root_scaled + stem_scaled
        # vertical stem lines connecting to text label
        ax.vlines(stem_lamb, ymin, ymax, **line_kwargs)

        # print text labels
        i = 0
        y_text = ymax + text_yoff_scaled
        text_objects = []

        for line in self.spectral_lines.values():
            font_dict = text_kwargs.copy()
            font_dict.update(line.plotting_style)

            for wavel in line.wavelengths:
                if xlim is not None:
                    cond = (max(wavel).value > xlim[0]) and (min(wavel).value < xlim[1])
                else:
                    cond = True

                if cond:
                    txt = ax.text(
                        stem_lamb[i],
                        y_text,
                        s=line.ion_name,
                        fontdict=font_dict,
                        clip_on=True,  # important
                    )
                    text_objects.append(txt)

                i += 1

        # --- force a draw so text positions are known ---
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # --- compute max text top in data coordinates ---
        ymax_text = -np.inf
        for txt in text_objects:
            bbox = txt.get_window_extent(renderer=renderer)
            bbox_data = bbox.transformed(ax.transData.inverted())
            ymax_text = max(ymax_text, bbox_data.y1)

        # --- adapt ylim if needed ---
        _, ymax = ax.get_ylim()
        if ymax < ymax_text:
            logger.warning(
                f"Text out of ylim, automatically adapting ymax now from "
                f"ymax_old={ymax:.2f} to ymax_new={1.05 * ymax_text:.2f}"
            )
            ax.set_ylim(None, 1.05 * ymax_text)

        return ax
