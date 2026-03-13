import abc
from astropy import units as u
from astropy.constants import k_B, m_p
from typing import List
from pathlib import Path
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .spectrum import Spectrum
from .utils.logging import setup_log
from .powr import (
    read_params_from_kasdefs,
    romans,
    read_elements_datom_header,
    powrsplinpo,
)

logger = setup_log(__name__)


class Model(object, metaclass=abc.ABCMeta):
    """
    should contain all information about model
    Only reads from model when specific things from model are requested, e.g. spectrum, parameters, etc.
    """

    def __init__(self, directory_name):
        self._directory = Path(directory_name)
        if self._directory.exists() == False:
            logger.error(f"{self._directory} does not exist")

    @property
    def directory(self):
        return self._directory


class PoWRModel(Model):
    def __init__(self, directory_name):
        super().__init__(directory_name)

        # self._params = self._read_params()

    def spectrum(
        self,
        keywords: List = [""],
        filename="formal.plot",
        name=None,
        vrad=0.0 * u.km / u.s,
    ):

        return self._read_spectrum(
            keywords=keywords,
            filename=filename,
            dataset=1,
            name=name,
            vrad=vrad,
        )

    def sed(
        self,
        with_lines=True,
        scale2distance=10 * u.pc,
        keywords: List = [""],
        name=None,
        vrad=0.0 * u.km / u.s,
    ):

        # in the formal.plot file, the SED with lines is in dataset 3, while the SED without lines is in dataset 2
        if with_lines:
            dataset = 3
        else:
            dataset = 2

        # read SED scaled at 10 pc from model, which is the default distance for SEDs in PoWR
        sp = self._read_spectrum(
            keywords=keywords,
            filename="formal.plot",
            dataset=dataset,
            x_unit=u.angstrom,
            y_unit=u.erg / (u.s * u.cm**2 * u.angstrom),
            name=name,
            vrad=vrad,
        )

        if scale2distance != 10 * u.pc:
            # scale spectrum to desired distance
            y = (sp.y * (scale2distance / (10 * u.pc)) ** 2).to(sp.y.unit)
            return Spectrum(
                x=sp.x, y=y, x_unit=sp.x.unit, y_unit=sp.y.unit, name=sp.name
            )
        else:
            return sp

    def _read_spectrum(
        self,
        keywords: List = [""],
        filename="formal.plot",
        dataset: int = 1,
        x_unit: u.Unit = None,
        y_unit: u.Unit = None,
        name=None,
        vrad=0.0 * u.km / u.s,
    ):
        file_spectrum = self._directory / Path(filename)
        if file_spectrum.exists():
            return Spectrum.from_powr(
                file_spectrum,
                keywords=keywords,
                dataset=dataset,
                x_unit=x_unit,
                y_unit=y_unit,
                name=name,
                vrad=vrad,
            )
        else:
            logger.error(f"{file_spectrum} does not exist")

    def _read_params(self, filename="modinfo.kasdefs"):
        file_params = self._directory / Path(filename)
        if file_params.exists() and file_params.suffix == ".kasdefs":
            return read_params_from_kasdefs(file_params)
        else:
            logger.error(
                f"{file_params} does not exist or is not a .kasdefs file. Use modinfo.kasdefs file."
            )

    @property
    def params(self):
        return self._read_params()

    @property
    def mass_fractions(self):
        mass_fractions = {}
        for key, value in self.params.items():
            if key.startswith("Xm_"):
                mass_fractions[key[3:]] = value
        return mass_fractions

    @property
    def number_fractions(self):
        number_fractions = {}
        for key, value in self.params.items():
            if key.startswith("Xn_"):
                number_fractions[key[3:]] = value
        return number_fractions

    def get_structure_data(self, structure_var: str) -> np.ndarray:
        """
        The 'msread' program is read with structure_var as input. The resulting text output is read and cut accordingly to only contain the numeral data.

        Parameters
        ----------
        structure_var : str
            Any parameter stored in the model file. See 'msinfo MODEL INFO-L' in a terminal for contents
        """

        home = os.getenv("HOME")

        msread = os.getenv(
            "~/powr/proc.dir/msread.com",
            default=f"{home}/powr/proc.dir/msread.com",
        )

        command = [f"{msread}", f"{structure_var}"]
        result = subprocess.run(
            command, cwd=self.directory, capture_output=True, text=True
        )

        if "msread error" in result.stdout:
            raise RuntimeError(f"Command failed: {result.stdout}")
        elif result.returncode != 0:
            raise RuntimeError(f"Command failed: {result.stderr}")

        structure_data = np.genfromtxt(
            result.stdout.splitlines(), skip_header=7, skip_footer=3
        )

        return structure_data

    @property
    def popnums(self):
        ND = int(self.get_structure_data("ND"))
        popnum_raw = self.get_structure_data("POPNUM")
        popnum_raw = popnum_raw.reshape(int(len(popnum_raw) / ND), ND)

        file_name = self._directory / "DATOM"
        with open(file_name, "r") as file:
            datom = np.array(
                list(map(str.strip, file.readlines()))
            )  # Reads all lines into a list

        start_string = "*KEYWORD--  ---NAME--- SYMB   ATMASS   STAGE"
        end_string = "*KEYWORD--UPPERLEVEL  LOWERLEVEL--EINSTEIN  RUD-CEY '--COLLISIONAL COEFFICIENTS--"

        start_lines = np.where(datom == start_string)[0]
        end_lines = np.where(datom == end_string)[0]

        levels = datom[start_lines[0] + 6 : end_lines[0]]
        for ii in range(1, start_lines.shape[0]):
            levels = np.append(levels, datom[start_lines[ii] + 6 : end_lines[ii]])

        # Based on Joris Josiek's script for the same use; modified for use in this class system.

        total_ions = np.sum(end_lines - (start_lines + 6))
        elements = np.full(total_ions, 10 * "a")
        ionisations = np.full(total_ions, 10 * "a")
        species = np.full(total_ions, 10 * "a")
        combined_popnum = {}

        for ii, level in enumerate(levels):
            level = level[12:22]

            # Split up element part and ionisation level part:
            if "P" not in level[:2].split()[0]:
                elements[ii] = level[:2].split()[0]
                ionisations[ii] = level[2:].lstrip()
            else:
                elements[ii] = level[:1].split()[0]
                ionisations[ii] = level[1:].lstrip()

            # Consistent naming using roman numerals:
            try:
                # If ionization is a simple integer
                ionisations[ii] = romans[int(ionisations[ii][0])]

            except ValueError:
                # Detect length of Roman number
                end = 0
                while ionisations[ii][end] in "IVX":
                    end += 1
                # Extract Roman number
                r = ionisations[ii][:end]
                # Weird exception if 'VX' appears in the level
                if r == "VX":
                    r = "V"
                ionisations[ii] = r

            # Attach standardised element and ionisation parts together again
            species[ii] = elements[ii] + ionisations[ii]

            # Combine level populations of the same ion
            if species[ii] in combined_popnum:
                combined_popnum[species[ii]] = (
                    combined_popnum[species[ii]] + popnum_raw[ii]
                )
            else:
                combined_popnum[species[ii]] = popnum_raw[ii]

        # iron_superlevels = {'G1':1, 'G2':3, 'G3':13, 'G4':18, 'G5':22, 'G6':29, 'G7':19, 'G8':14, 'G9':15}
        iron_superlevels = {
            "G2": 1,
            "G3": 13,
            "G4": 18,
            "G5": 22,
            "G6": 29,
            "G7": 1,
        }  # G2 - G7
        iron_popnums = {}

        level_sum = 0
        for ii, el in enumerate(iron_superlevels):

            index0 = len(levels) + level_sum
            index1 = index0 + iron_superlevels[el]

            if iron_superlevels[el] == 1:
                iron_popnums[el] = popnum_raw[index0:index1, :][0]
            elif iron_superlevels[el] > 1:
                iron_popnums[el] = np.sum(popnum_raw[index0 : index1 - 1, :], axis=0)
            level_sum += iron_superlevels[el]

        popnums = {}
        popnums.update(combined_popnum)
        popnums.update(iron_popnums)

        return popnums

    @property
    def radiative_acceleration_ions(self):
        # read all ions and elems
        elems = read_elements_datom_header(self._directory / "DATOM")

        # Read and split the acceleration per ion
        ND = int(self.get_structure_data("ND"))
        Nels = self.get_structure_data("ABXYZ").shape[0]

        aradion = self.get_structure_data("ARADION")
        aradion = aradion.reshape(27, Nels, ND - 1)
        return elems, aradion

    @property
    def radiative_acceleration_continuum(self):
        ND = int(self.get_structure_data("ND"))
        Nels = self.get_structure_data("ABXYZ").shape[0]

        acntelem = self.get_structure_data("ACNTELEM")
        acntelem = acntelem.reshape(ND - 1, Nels)

        return acntelem

    @property
    def a_pressure(self):
        """Compute mechanical acceleration due to pressure gradient using PoWR's powrsplinpo.

        Returns:
            apress : ndarray, acceleration in cm/s^2 at midpoints (length len(R)-1)"""
        # Attach units
        rstar = self.params["RSTAR"] * u.cm
        R = np.array(self.get_structure_data("R")) * u.dimensionless_unscaled
        xmu = np.array(self.params["XMU"]) * u.dimensionless_unscaled
        rho = np.array(self.get_structure_data("RHO")) * u.g / u.cm**3
        vturb = np.array(self.params["VTURB"]) * u.km / u.s
        T = np.array(self.get_structure_data("T")) * u.K

        # Convert radius to physical units
        RRstar = R * rstar

        # Sound speed
        asound = np.sqrt((T * k_B / (xmu * m_p)).to(u.cm**2 / u.s**2))

        # Total pressure: thermal + turbulent
        press = rho * (asound**2 + vturb.to(u.cm / u.s) ** 2)

        # Depth points and midpoints
        Ndp = len(R)
        dp = np.arange(1, Ndp + 1)
        dp_mid = 0.5 * (dp[:-1] + dp[1:])

        # Interpolate radius and density at midpoints
        RRstar_mid = np.interp(dp_mid, dp, RRstar.value)
        rho_mid = np.interp(dp_mid, dp, rho.value) * rho.unit

        # Compute dP/dr at midpoints using PoWR's powrsplinpo
        Pdr = []
        for r_mid in RRstar_mid:
            # PoWR.powrsplinpo returns tuple: (pressure_spline, derivative)
            Pdr.append(powrsplinpo(r_mid, press.value, RRstar.value, True)[1])
        dPdr = np.array(Pdr) * (
            press.unit / RRstar.unit
        )  # convert derivative to physical units

        # Acceleration
        apress = -(dPdr / rho_mid)  # in cm/s^2

        return apress.to(u.cm / u.s**2).value  # array of length Ndp-1

    @property
    def a_mechanical(self):
        v = self.get_structure_data("VELO") * u.km / u.s
        r = self.get_structure_data("R") * self.params["RSTAR"] * u.cm

        # Compute velocity at midpoints
        v_mid = 0.5 * (v[:-1] + v[1:])

        # Compute dv/dr using finite differences
        dv = v[1:] - v[:-1]  # length N-1
        dr = r[1:] - r[:-1]  # length N-1
        vdvdr = (v_mid * dv / dr).to(u.cm / u.s**2).value  # derivative at shell centers
        return vdvdr

    def plot_popnums(
        self,
        element,
        ax=None,
        text_in_legend=False,
        use_marker=False,
        **kwargs,
    ):
        """Plot population numbers

        :param element: determine for which element(s) the population numbers should be plotted. If a string of an element is given, e.g. 'C', then all ionisation stages of that element will be plotted, e.g. 'CIII' and 'CII'. If a list of ionisation stages is given, e.g. ['CIII', 'CII'], then only those ionisation stages will be plotted. If a single string of an ionisation stage is given, e.g. 'CIII', then only that ionisation stage will be plotted.
        :type element: list or str
        :param ax: matplotlib axis, defaults to None
        :type ax: matplotlib.axis, optional
        :param text_in_legend: whether to include text in legend, defaults to False
        :type text_in_legend: bool, optional
        :param use_marker: whether to use markers , defaults to False
        :type use_marker: bool, optional
        :return: matplotlib axis
        :rtype: matplotlib.axis
        """
        markers = ["o", "v", "*", "s", "<", "D", "^", "+", "p", ">", "h"]
        n_tot = self.get_structure_data("ENTOT")
        popnums = self.popnums

        if ax is None:
            fig, ax = plt.subplots()

        # If all ionisation levels of a specific element should be plotted, e.g. 'C', then plot the population numbers of all ionisation stages of that element, e.g. 'CIII' and 'CII'
        if type(element) == str:
            # if element is an ionisation stage, e.g. 'CIII', then only plot the population numbers of that ionisation stage
            if element in popnums:
                if ax is None:
                    fig, ax = plt.subplots()

                if use_marker:
                    kwargs["marker"] = "o"
                    kwargs["linestyle"] = "None"
                else:
                    kwargs["linestyle"] = "-"
                if text_in_legend:
                    kwargs["label"] = element
                else:
                    t = ax.text(
                        np.log10(n_tot[0]),
                        np.log10(popnums[element][0]),
                        element,
                        color=kwargs.get("color", "black"),
                        verticalalignment="center",
                        horizontalalignment="center",
                    )
                    t.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="none"))

                ax.plot(np.log10(n_tot), np.log10(popnums[element]), **kwargs)
                ax.legend()
                ax.set_xlabel(r"$\log (n_{\mathrm{tot}} / \mathrm{cm}^{-3})$")
                ax.set_ylabel(r"$\log (n_i / n_{\mathrm{tot}})$")
                ax.set_ylabel("Population number")
            else:

                j = 0
                if use_marker:
                    kwargs["linestyle"] = "None"
                for key, value in popnums.items():
                    # the key is of the form 'FeIII' and we want to only plot the population numbers of the ionisation stage of the given element, e.g. 'CIII' and 'CII' for element 'C'
                    # keep in mind that there are sometimes multiple elements with the same start 'HEI' and 'HI'
                    # So, always check if next character is a I, V, X or a number, which are the possible ionisation stages in PoWR models
                    if (key.startswith(element)) and (
                        key[len(element) : len(element) + 1] in ["I", "V", "X"]
                    ):
                        # + [str(i) for i in range(10)]:
                        j += 1
                        if use_marker:
                            kwargs["marker"] = markers[j % len(markers)]
                            kwargs["linestyle"] = "None"
                        if text_in_legend:
                            kwargs["label"] = key
                        else:
                            t = ax.text(
                                np.log10(n_tot[0]),
                                np.log10(popnums[key][0]),
                                key,
                                color=kwargs.get("color", "black"),
                                verticalalignment="center",
                                horizontalalignment="center",
                            )
                            t.set_bbox(
                                dict(facecolor="white", alpha=0.7, edgecolor="none")
                            )

                        ax.plot(
                            np.log10(n_tot),
                            np.log10(popnums[key]),
                            **kwargs,
                        )
        # If element is a list of ionisation stages, e.g. ['CIII', 'CII'], then plot the population numbers of those ionisation stages
        elif type(element) == list:
            for j, el in enumerate(element):
                if el in popnums:
                    if text_in_legend:
                        kwargs["label"] = el
                    else:

                        t = ax.text(
                            np.log10(n_tot[0]),
                            np.log10(popnums[el][0]),
                            el,
                            color=kwargs.get("color", "black"),
                            verticalalignment="center",
                            horizontalalignment="center",
                        )
                        t.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="none"))

                    if use_marker:
                        kwargs["marker"] = markers[j % len(markers)]
                        kwargs["linestyle"] = "None"

                    ax.plot(
                        np.log10(n_tot),
                        np.log10(popnums[el]),
                        **kwargs,
                    )
                else:
                    logger.warning(f"{el} not found in population numbers")

        else:
            logger.error(
                f"Element should be either a string, e.g. 'C', or a list of ionisation stages, e.g. ['CIII', 'CII'], or a single string of an ionisation stage, e.g. 'CIII', but got {type(element)}"
            )

        if text_in_legend:
            ax.legend()

        ax.set_xlabel(r"$\log (n_{\mathrm{tot}} / \mathrm{cm}^{-3})$")
        ax.set_ylabel(r"$\log (n_i / n_{\mathrm{tot}})$")
        return ax

    def plot_radiative_acceleration_per_ion(
        self,
        ax=None,
        all_ions_in_one_plot=False,
        n_cols_legend=1,
        rename_G2Fe=True,
        **kwargs,
    ):
        """Plot radiative acceleration per ion

        :param ax: matplotlib axis, only used if all_ions_in_one_plot is True, defaults to None
        :type ax: matplotlib.axis, optional
        :param all_ions_in_one_plot: determine whether to plot all ion acceleration contributions in one plot, defaults to False
        :type all_ions_in_one_plot: bool, optional
        :param n_cols_legend: number of columns in the legend, defaults to 1
        :type n_cols_legend: int, optional
        :param rename_G2Fe: whether to rename G to Fe, defaults to True
        :type rename_G2Fe: bool, optional
        :return: matplotlib axis or list of axes if all_ions_in_one_plot is False
        :rtype: matplotlib.axis or list of matplotlib.axis
        """
        markers = ["o", "v", "*", "s", "<", "D", "^", "+", "p", ">", "h"]
        if all_ions_in_one_plot:
            if ax is None:
                fig, ax = plt.subplots()
        else:
            axes = []
            if ax is not None:
                logger.warning(
                    "ax is provided but all_ions_in_one_plot is False. Ignoring ax and creating new subplots for each ion."
                )
        elems, a = self.radiative_acceleration_ions

        r = self.get_structure_data("R")
        r_mid = 0.5 * (r[1:] + r[:-1])

        # logg10 of radius dependent gravity in cm/s^2 at midpoints
        logg_r = np.log10(
            10 ** self.params["GLOG"] / (self.get_structure_data("R") ** 2)
        )
        logg_mid = 0.5 * (logg_r[:-1] + logg_r[1:])

        j = 0
        for elem, ions in elems.items():
            if all_ions_in_one_plot == False:
                fig, ax = plt.subplots()
                axes.append(ax)

            if rename_G2Fe and elem == "G":
                if elem == "G":
                    elem = "Fe"

            for ion in ions:
                acc = a[ion - 1, j]
                if all_ions_in_one_plot:
                    color = f"C{j}"
                    marker = markers[ion - 1 % len(markers)]

                else:
                    color = None
                    marker = kwargs.pop("marker", None)

                ax.plot(
                    np.log10(r_mid - 1),
                    np.log10(acc) - logg_mid,
                    label=f"{elem[0].upper() + elem[1:].lower() if len(elem) > 1 else elem} {romans[ion]}",
                    color=color,
                    marker=marker,
                    **kwargs,
                )
            j += 1

        def plot_other_accelerations(ax, **kwargs):
            accelerations = [
                (self.get_structure_data("ARAD"), r"$a_\mathrm{rad}$", "C00"),
                (self.a_pressure, r"$a_\mathrm{press}$", "C01"),
                (self.get_structure_data("ATHOM"), r"$a_\mathrm{thom}$", "C02"),
            ]

            for acc, label, color in accelerations:
                ax.plot(
                    np.log10(r_mid - 1),
                    np.log10(acc) - logg_mid,
                    label=label,
                    color=color,
                    linestyle="--",
                    **kwargs,
                )

            return ax

        if all_ions_in_one_plot:
            plot_other_accelerations(ax, **kwargs)
            ax.legend(ncol=n_cols_legend)
            ax.set_xlabel(r"$\log_{10} (r / R_* - 1)$")
            ax.set_ylabel(r"$\log_{10} (a / g)$")
            return ax
        else:
            for ax in axes:
                plot_other_accelerations(ax, **kwargs)
                ax.legend(ncol=n_cols_legend)
                ax.set_xlabel(r"$\log_{10} (r / R_* - 1)$")
                ax.set_ylabel(r"$\log_{10} (a / g)$")
            return axes

    def plot_velocity_stratification(self, ax=None, **kwargs):
        """Plot velocity stratification (velocity vs radius)

        :param ax: matplotlib axis, defaults to None
        :type ax: matplotlib.axis, optional
        :return: matplotlib axis
        :rtype: matplotlib.axis
        """
        if ax is None:
            fig, ax = plt.subplots()

        v = self.get_structure_data("VELO")
        r = self.get_structure_data("R")
        mask = r != 1.0
        r = r[mask]
        v = v[mask]
        ax.plot(np.log10(r - 1), v / 100, **kwargs)
        ax.set_xlabel(r"$\log_{10} (r / R_* - 1)$")
        ax.set_ylabel(r"$v(r) [100 \, \mathrm{km/s}]$")
        return ax

    def plot_acceleration_vs_velocity(self, ax=None, print_work_ratio=True, **kwargs):
        """Plot all acceleration terms as function of the velocity

        :param ax: matplotlib axis, defaults to None
        :type ax: matplotlib.axis, optional
        :param print_work_ratio: whether to print work ratio, defaults to True
        :type print_Q: bool, optional
        :return: matplotlib axis
        :rtype: matplotlib.axis
        """
        if ax is None:
            fig, ax = plt.subplots()

        v = self.get_structure_data("VELO") * u.km / u.s
        v_mid = 0.5 * (v[:-1] + v[1:])
        v_vinf = v_mid / v[0]

        # logg10 of radius dependent gravity in cm/s^2 at midpoints
        logg_r = np.log10(
            10 ** self.params["GLOG"] / (self.get_structure_data("R") ** 2)
        )
        logg_mid = 0.5 * (logg_r[:-1] + logg_r[1:])

        apress = np.log10(self.a_pressure) - logg_mid
        amechanical = np.log10(self.a_mechanical) - logg_mid
        arad = np.log10(self.get_structure_data("ARAD")) - logg_mid
        acont = np.log10(self.get_structure_data("ACONT")) - logg_mid
        athom = np.log10(self.get_structure_data("ATHOM")) - logg_mid

        a_rad_plus_press = np.log10(apress + self.get_structure_data("ARAD")) - logg_mid
        g_plus_amech = np.log10(amechanical + 10**logg_mid) - logg_mid

        ax.set_ylim(-3, None)
        ax.set_xlim(0, np.max(v_vinf))

        # Mark the sonic point and the point where the continuum acceleration equals gravity with vertical dashed lines
        Rsonic = self.params["RSONIC"]
        idx = np.abs(self.get_structure_data("R") - Rsonic).argmin()

        Rcon = self.params["RCON"]
        idx_con = np.abs(self.get_structure_data("R") - Rcon).argmin()

        ax.axvline(x=v_vinf[idx], color="0.4", linestyle="--")
        t = ax.text(
            v_vinf[idx],
            0.92 * ax.get_ylim()[1],
            "$R_{\mathrm{sonic}}$",
            rotation=90,
            color="grey",
            verticalalignment="top",
            horizontalalignment="center",
        )
        t.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.axvline(x=v_vinf[idx_con], color="0.4", linestyle="--")
        t = ax.text(
            v_vinf[idx_con],
            0.92 * ax.get_ylim()[1],
            "$R_{\mathrm{con}}$",
            rotation=90,
            color="grey",
            verticalalignment="top",
            horizontalalignment="center",
        )
        t.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.plot(v_vinf, g_plus_amech, label="$(g+a_{\mathrm{mech}})/g$", **kwargs)
        ax.plot(v_vinf, arad, label="$a_{\mathrm{rad}}/g$", **kwargs)
        ax.plot(v_vinf, amechanical, label="$a_{\mathrm{mech}}/g$", **kwargs)
        ax.plot(v_vinf, acont, label="$a_{\mathrm{cont}}/g$", **kwargs)
        ax.plot(
            v_vinf,
            a_rad_plus_press,
            label="$(a_{\mathrm{rad}}+a_{\mathrm{press}})/g$",
            **kwargs,
        )
        ax.plot(v_vinf, apress, label="$a_{\mathrm{press}}/g$", **kwargs)
        ax.plot(v_vinf, athom, label="$a_{\mathrm{thom}}/g$", **kwargs)

        ax.legend()

        if print_work_ratio:

            # Add extra text entry
            extra_text = mpatches.Patch(
                color="none", label=f"Q = {self.params['WORKRATIO']}"
            )

            # Retrieve current handles and labels
            handles, labels = ax.get_legend_handles_labels()

            # Append the new text entry
            handles.insert(0, extra_text)
            labels.insert(0, extra_text.get_label())

            # Update legend
            ax.legend(handles=handles, labels=labels, loc="upper left")

        ax.set_xlabel("$v(r)/v_\infty$")
        ax.set_ylabel("$\log_{10}(a/g)$")

        return ax

    def plot_acceleration_vs_depth_index(
        self, ax=None, print_work_ratio=True, **kwargs
    ):
        """Plot all acceleration terms as function of the depth index,

        :param ax: matplotlib axis, defaults to None
        :type ax: matplotlib.axis, optional
        :param print_work_ratio: whether to print work ratio in legend, defaults to True
        :type print_work_ratio: bool, optional
        :return: matplotlib axis
        :rtype: matplotlib.axis
        """
        if ax is None:
            fig, ax = plt.subplots()

        depth_index = np.arange(1, len(self.get_structure_data("R")))

        # logg10 of radius dependent gravity in cm/s^2 at midpoints
        logg_r = np.log10(
            10 ** self.params["GLOG"] / (self.get_structure_data("R") ** 2)
        )
        logg_mid = 0.5 * (logg_r[:-1] + logg_r[1:])

        apress = np.log10(self.a_pressure) - logg_mid
        amechanical = np.log10(self.a_mechanical) - logg_mid
        arad = np.log10(self.get_structure_data("ARAD")) - logg_mid
        acont = np.log10(self.get_structure_data("ACONT")) - logg_mid
        athom = np.log10(self.get_structure_data("ATHOM")) - logg_mid

        a_rad_plus_press = (
            np.log10(self.a_pressure + self.get_structure_data("ARAD")) - logg_mid
        )
        g_plus_amech = np.log10(amechanical + 10**logg_mid) - logg_mid

        ax.set_ylim(-3, None)
        ax.set_xlim(0, np.max(depth_index))

        # Mark the sonic point and the point where the continuum acceleration equals gravity with vertical dashed lines
        Rsonic = self.params["RSONIC"]
        idx = np.abs(self.get_structure_data("R") - Rsonic).argmin()

        Rcon = self.params["RCON"]
        idx_con = np.abs(self.get_structure_data("R") - Rcon).argmin()

        ax.axvline(x=depth_index[idx], color="0.4", linestyle="--")
        t = ax.text(
            depth_index[idx],
            0.92 * ax.get_ylim()[1],
            "$R_{\mathrm{sonic}}$",
            rotation=90,
            color="grey",
            verticalalignment="top",
            horizontalalignment="center",
        )
        t.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.axvline(x=depth_index[idx_con], color="0.4", linestyle="--")
        t = ax.text(
            depth_index[idx_con],
            0.92 * ax.get_ylim()[1],
            "$R_{\mathrm{con}}$",
            rotation=90,
            color="grey",
            verticalalignment="top",
            horizontalalignment="center",
        )
        t.set_bbox(dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.plot(depth_index, g_plus_amech, label="$(g+a_{\mathrm{mech}})/g$", **kwargs)
        ax.plot(depth_index, arad, label="$a_{\mathrm{rad}}/g$", **kwargs)
        ax.plot(depth_index, amechanical, label="$a_{\mathrm{mech}}/g$", **kwargs)
        ax.plot(depth_index, acont, label="$a_{\mathrm{cont}}/g$", **kwargs)
        ax.plot(
            depth_index,
            a_rad_plus_press,
            label="$(a_{\mathrm{rad}}+a_{\mathrm{press}})/g$",
            **kwargs,
        )
        ax.plot(depth_index, apress, label="$a_{\mathrm{press}}/g$", **kwargs)
        ax.plot(depth_index, athom, label="$a_{\mathrm{thom}}/g$", **kwargs)

        ax.legend()

        if print_work_ratio:

            # Add extra text entry
            extra_text = mpatches.Patch(
                color="none", label=f"Q = {self.params['WORKRATIO']}"
            )

            # Retrieve current handles and labels
            handles, labels = ax.get_legend_handles_labels()

            # Append the new text entry
            handles.insert(0, extra_text)
            labels.insert(0, extra_text.get_label())

            # Update legend
            ax.legend(handles=handles, labels=labels, loc="upper left")

        ax.set_xlabel("Depth Index L")
        ax.set_ylabel("$\log_{10}(a/g)$")

        return ax
