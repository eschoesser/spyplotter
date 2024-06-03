from pathlib import Path
import pkg_resources


def get_path_of_config_file(config_file) -> Path:
    file_path = pkg_resources.resource_filename("spyplotter", "config/%s" % config_file)
    return file_path
