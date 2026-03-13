from pathlib import Path
from importlib import resources
import matplotlib.pyplot as plt
import os


def get_path_of_config_file(config_file) -> Path:
    """Return the path to a config file inside spyplotter/config."""
    return resources.files("spyplotter").joinpath("config", config_file)


def get_path_of_data_file(data_file) -> Path:
    """Return the path to a data file inside spyplotter/data."""
    return resources.files("spyplotter").joinpath("data", data_file)


def load_y_unit(y_unit_file="y_unit"):
    """Loads the custom y_unit settings."""
    y_unit_path = get_path_of_config_file(y_unit_file)
    if os.path.exists(y_unit_path):
        plt.rcParams.update(plt.matplotlib.rc_params_from_file(y_unit_path))
