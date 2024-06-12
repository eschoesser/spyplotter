from pathlib import Path
import pkg_resources


def get_path_of_config_file(config_file) -> Path:
    file_path = pkg_resources.resource_filename("spyplotter", f"config/{config_file}")
    return file_path


def get_path_of_data_file(data_file) -> Path:
    file_path = pkg_resources.resource_filename("spyplotter", "data/%s" % data_file)
    return file_path
