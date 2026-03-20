from .spectrum import Spectrum
from .line_identification import LineIdentifier, SpectralLine
from .utils.logging import setup_log
from .utils.package_data import get_path_of_config_file
from .model import PoWRModel
from .ism_lines import ISMModel, HLymanA

from spyplotter.utils.logging import update_logging_level


from .utils.package_data import load_matplotlibrc

# configuration matplotlib
# from matplotlib import rc_file,rcdefaults
import matplotlib.pyplot as plt

load_matplotlibrc()
