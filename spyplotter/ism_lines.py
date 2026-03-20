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

import astropy.constants as c

from scipy.special import wofz

logger = setup_log(__name__)

from .spec_tools.unit_checks import (
    check_x_unit,
    check_velocity_unit,
    check_column_density_unit,
    check_T_unit,
)

from .line_identification import LineIdentifier
from .spectrum import Spectrum


class ISMLine:
    def __init__(self, lam0, f, gamma):
        self.lam0 = check_x_unit(lam0)
        self.f = f
        if isinstance(
            gamma, (u.quantity.Quantity, SpectralCoord, SpectralQuantity)
        ) and gamma.unit.is_equivalent(1 / u.s):
            logger.debug(f"Use given gamma unit: {gamma.unit}")
            self.gamma = gamma.to(1 / u.s)
        elif isinstance(gamma, (float, int)):
            logger.info("No unit for gamma specified. Thus assuming 1/s.")
            self.gamma = gamma * (1 / u.s)
        else:
            logger.error(
                f"Not known format for gamma used: {gamma}. Convert to float or astropy classes Quantity, SpectralCoord or SpectralQuantity with unit 1/s"
            )

    def tau(self, wavelength, N, T, vturb, A, vrad):
        lam = check_x_unit(wavelength)

        # electric charge in cgs
        e = c.e.esu

        # shift rest wavelength
        lam0 = self.lam0 * (1 + (vrad / c.c).decompose())

        # decide profile
        use_voigt = (T > 0 * u.K) or (vturb > 0 * u.cm / u.s)
        dlam = lam - lam0

        if use_voigt:
            # Doppler parameter
            if (A > 0) and (T > 0 * u.K):
                b = np.sqrt(2 * c.k_B * T / (c.m_p * A) + vturb**2)
            else:
                b = vturb.to(u.cm / u.s)

            dopp = lam0 / c.c * b
            # Coefficients for the Voigt function
            a = (0.25 * self.gamma / (np.pi * dopp) * lam0**2 / c.c).decompose()
            v = (np.abs(dlam) / dopp).decompose()
            # Normalized voigt function
            H = np.real(wofz(v + 1j * a)) / np.sqrt(np.pi)
            frb = (np.pi * e**2 / (c.m_e * c.c)).decompose()
            tau = (frb * self.f * N * H * lam0**2 / (dopp * c.c)).decompose()
            return tau.value

        else:
            # use Lorentz profile
            c0 = (4 * np.pi * e**2 / (c.m_e * c.c)) * self.f / self.gamma
            c1 = (((4 * np.pi * c.c) / (lam**2 * self.gamma)) ** 2).decompose()
            tau = (c0 * N / (1 + c1 * dlam**2)).decompose()
            return tau.value

    @property
    def x_unit(self):
        return self.lam0.unit


class ISMIon:
    def __init__(self, ion_dict, ion_name):
        self.ion_name = ion_name

        gas = ion_dict["gas"]

        self.N = check_column_density_unit(gas["n"])
        self.A = gas.get("A", 0)

        self.T = check_T_unit(gas.get("T", 0 * u.K))
        self.vturb = check_velocity_unit(gas.get("vturb", 0 * u.km / u.s))
        self.vrad = check_velocity_unit(gas.get("vrad", 0 * u.km / u.s))

        self.lines = {
            name: ISMLine(l["lambda"], l["f"], l["gamma"])
            for name, l in ion_dict["lines"].items()
        }

    @property
    def wavelengths(self):
        return [line.lam0 for line in self.lines.values()]

    @property
    def wavelength_range(self):
        lam0s = self.wavelengths
        return min(lam0s), max(lam0s)

    @property
    def x_unit(self):
        return self.wavelength_range[0].unit

    def tau(self, wavelength):
        tau_total = np.zeros_like(wavelength.value)

        for line in self.lines.values():
            tau_total += line.tau(
                wavelength, self.N, self.T, self.vturb, self.A, self.vrad
            )

        return tau_total

    def to_LineIdentifier(self, suffix_line_name=""):
        spectral_lines_dict = {
            self.ion_name
            + f" {suffix_line_name}": {
                "wavelengths": [[line.lam0.value] for line in self.lines.values()]
            }
        }
        return LineIdentifier.from_dict(
            spectral_lines_dict, x_unit=self.x_unit, vrad=self.vrad
        )

    def plot(self, sp=None, ax=None, **kwargs):
        sp_ism = self.to_spectrum(sp=sp)

        if ax is None:
            fig, ax = plt.subplots()

        ax = sp_ism.plot(ax=ax, **kwargs)

        return ax

    def to_spectrum(self, sp=None):
        if sp is not None:
            x = sp.x
            y = sp.y * np.exp(-self.tau(x))
        else:
            x_range = self.wavelength_range
            x = np.linspace(0.8 * x_range[0], 1.2 * x_range[1], 1000)
            y = np.exp(-self.tau(x))

        return Spectrum(x, y)


class HLymanA(ISMIon):
    def __init__(
        self,
        ebv=None,
        n_HI=None,
        T=0 * u.K,
        vturb=0 * u.km / u.s,
        vrad=0 * u.km / u.s,
    ):
        """
        Hydrogen Lyman series absorber (Groenewegen & Lamers 1989)
        """

        # --- column density conversion from wrplot---
        # C2 = 10. ** 21.58 = HYDROGEN COLUMN DENSITY FOR E(B-V)= 1 MAG
        # ACHTUNG: BOHLIN ET AL. GEBEN EINEN GROESSEREN WERT: 5.8E21
        C2 = 3.8e21 / u.cm**2  # Fortran value

        if n_HI is not None:
            N = n_HI
        elif ebv is not None:
            N = ebv * C2
        else:
            raise ValueError("Provide either EBV or n_HI")

        # The following data is directly copied from the wrplot source code
        """ These data cover the series up to n=10. An extension up to n=20 has been added by Thomas Rauch (see WR-Memo 19-Apr-2007). He writes: 
            "Ausrechnen (der atomaren Querschnitte) ist wohl ein schwieriges
            Geschaeft, da man de f-Werte aller s-p- und d-p-Uebergaenge vom HI dazu
            benoetigt - keine Ahnung wo ich die nun herbekommen sollte. Deshalb habe
            ich die C0 und C1 mal logarithmisch aufgetragen und dann nach Augenmass"""

        lam0 = (
            np.array(
                [
                    1215.671,
                    1025.722,
                    972.5366,
                    949.74297,
                    937.80344,
                    930.74824,
                    926.22571,
                    923.15037,
                    920.96313,
                    919.35146,
                    918.12939,
                    917.18061,
                    916.42917,
                    915.82388,
                    915.32904,
                    914.91936,
                    914.57629,
                    914.28622,
                    914.03860,
                    913.82569,
                    913.64126,
                    913.48036,
                    913.33922,
                    913.21470,
                    913.10431,
                    913.00597,
                    912.91800,
                    912.83898,
                    912.76775,
                    912.70331,
                    912.64482,
                ]
            )
            * u.AA
        )

        f = np.array(
            [
                4.1620e-01,
                7.9100e-02,
                2.8990e-02,
                1.3940e-02,
                7.7990e-03,
                4.8140e-03,
                3.1830e-03,
                2.2160e-03,
                1.6050e-03,
                1.2010e-03,
                9.2140e-04,
                7.2270e-04,
                5.7740e-04,
                4.6860e-04,
                3.8560e-04,
                3.2110e-04,
                2.7030e-04,
                2.2960e-04,
                1.9670e-04,
                1.6980e-04,
                1.4760e-04,
                1.2910e-04,
                1.1360e-04,
                1.0050e-04,
                8.9290e-05,
                7.9700e-05,
                7.1450e-05,
                6.4300e-05,
                5.8070e-05,
                5.2610e-05,
                4.7830e-05,
            ]
        )

        gamma = np.array(
            [
                6.26500e08,
                1.89700e08,
                8.12700e07,
                4.20400e07,
                2.45000e07,
                1.4043e07,
                9.3788e06,
                6.5742e06,
                4.7849e06,
                3.5931e06,
                2.7643e06,
                2.1728e06,
                1.7388e06,
                1.4131e06,
                1.1641e06,
                9.7030e05,
                8.1739e05,
                6.9477e05,
                5.9556e05,
                5.1437e05,
                4.4730e05,
                3.9138e05,
                3.4449e05,
                3.0485e05,
                2.7091e05,
                2.4188e05,
                2.1689e05,
                1.9522e05,
                1.7632e05,
                1.5978e05,
                1.4527e05,
            ]
        ) * (1 / u.s)

        lines = {}
        for i in range(len(lam0)):
            lines[f"Ly{i+1}"] = {"lambda": lam0[i], "f": f[i], "gamma": gamma[i]}

        ion_dict = {
            "gas": {
                "n": N,
                "A": 1.0,  # hydrogen
                "T": T,
                "vturb": vturb,
                "vrad": vrad,
            },
            "lines": lines,
        }

        super().__init__(ion_dict, ion_name="HI")


class ISMModel:
    def __init__(self, ism_dict):
        self.ions = {
            ion_name: ISMIon(ion_dict, ion_name)
            for ion_name, ion_dict in ism_dict.items()
        }

    def flux(self, wavelength):
        tau_total = np.zeros_like(wavelength.value)

        for ion in self.ions.values():
            tau_total += ion.tau(wavelength)

        return np.exp(-tau_total)  # keep your convention

    @property
    def x_unit(self):
        return list(self.ions.values())[0].x_unit

    @property
    def wavelengths(self):
        # flattened array of the wavelengths of all ISM lines
        wavelengths = []
        for ion in self.ions.values():
            wavelengths += ion.wavelengths
        wavelengths.sort()
        return wavelengths

    @property
    def wavelength_range(self):
        # min and max of all lines
        lams = self.wavelengths
        return min(lams), max(lams)

    def to_dict(self):
        return {
            name: {
                "gas": {
                    "n": float(ion.N.to(u.cm ** (-2)).value),
                    "T": float(ion.T.to(u.K).value),
                    "vturb": float(ion.vturb.to(u.km / u.s).value),
                    "A": float(ion.A),
                    "vrad": float(ion.vrad.to(u.km / u.s).value),
                },
                "lines": {
                    line_name: {
                        "lambda": float(line.lam0.to(u.AA).value),
                        "f": float(line.f),
                        "gamma": float(line.gamma.to(1 / u.s).value),
                    }
                    for line_name, line in ion.lines.items()
                },
            }
            for name, ion in self.ions.items()
        }

    def to_LineIdentifier(self, suffix_line_name="", vrad=0 * u.km / u.s):
        spectral_lines_dict = {
            name
            + f" {suffix_line_name}": {
                "wavelengths": [[line.lam0.value] for line in ion.lines.values()]
            }
            for name, ion in self.ions.items()
        }
        return LineIdentifier.from_dict(
            spectral_lines_dict, x_unit=self.x_unit, vrad=vrad
        )

    def to_yaml(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, "r") as f:
            ism_dict = yaml.safe_load(f)
        return cls(ism_dict)

    def plot(self, ax=None, sp=None, **kwargs):
        sp_ism = self.to_spectrum(sp=sp)

        ax = sp_ism.plot(ax=ax, **kwargs)
        return ax

    def to_spectrum(self, sp=None):
        if sp is not None:
            x = sp.x
            y = sp.y * self.flux(x)
        else:
            x_range = self.wavelength_range
            x = np.linspace(0.8 * x_range[0], 1.2 * x_range[1], 50000)
            y = self.flux(x)
        return Spectrum(x, y)
