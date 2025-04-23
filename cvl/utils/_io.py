import os
import yaml
import pickle as pk
from box import ConfigBox


def write_pickle_file(save_path, data):
    """Saves data to a pickle file """

    if os.sep in save_path:
        dir_path, file_name = os.path.split(save_path)
        os.makedirs(dir_path, exist_ok=True)

    with open(save_path, 'wb') as handle:
        pk.dump(data, handle, protocol=pk.HIGHEST_PROTOCOL)


def read_pickle_file(path):
    """Loads a pickle file """

    with open(path, 'rb') as handle:
        data = pk.load(handle)

    return data


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def read_yaml_config_file(file_path: str) -> ConfigBox:
    with open(file_path, 'r') as stream:
        cfgs = yaml.safe_load(stream)

    cfgs = ConfigBox(cfgs)

    return cfgs
