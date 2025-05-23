from pathlib import Path
import pkg_resources
import matplotlib.pyplot as plt
import os


def get_path_of_config_file(config_file) -> Path:
    file_path = pkg_resources.resource_filename("spyplotter", f"config/{config_file}")
    return file_path


def get_path_of_data_file(data_file) -> Path:
    file_path = pkg_resources.resource_filename("spyplotter", "data/%s" % data_file)
    return file_path


def load_matplotlibrc():
    """Loads the custom matplotlibrc settings."""
    matplotlibrc_path = get_path_of_config_file("matplotlibrc")
    if os.path.exists(matplotlibrc_path):
        plt.rcParams.update(plt.matplotlib.rc_params_from_file(matplotlibrc_path))
