from .spectrum import Spectrum
from .line_identification import LineIdentifier, SpectralLine
from .utils.logging import setup_log
from .utils.package_data import get_path_of_config_file
from .model import PoWRModel

from spyplotter.utils.logging import update_logging_level


# configuration matplotlib
from matplotlib import rc_file

rc_file(get_path_of_config_file("matplotlibrc_wrplotlike"))
