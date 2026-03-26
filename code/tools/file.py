import h5py
import yaml
import os 
import numpy as np
from datetime import datetime
from pathlib import Path

from tools.parameter import PATH_TO_CONFIG_FOLDER, PATH_TO_DATA_FOLDER, EQUAL_CONFIGS_EXCLUDE_KEY, COMPATIBLE_DICT_PARAMETER, DATA_BASE_NAME, DATA_SET_ADDITIVE_KEYS, DATA_SET_PARAMETER_KEYS 

# Configs
def write_config(configuration: dict, filepath: str, backup: bool = False):
    if backup:
        filepath = os.path.join(PATH_TO_CONFIG_FOLDER, filepath+".yaml")
    with open(filepath,"w") as file:
        yaml.dump(configuration, file)
    pass


def read_config(filepath: str):
    with open(filepath, "r") as file:
        configuration = yaml.safe_load(file)
    return configuration 


def compare_configs(config_1, config_2):
    # all entries identical 
    identical = config_1 == config_2 

    # all entries expect num_shots identical
    equal = (
        {k: v for k, v in config_1.items() if k != EQUAL_CONFIGS_EXCLUDE_KEY}
        ==
        {k: v for k, v in config_2.items() if k != EQUAL_CONFIGS_EXCLUDE_KEY})

    # all entries that do not show up in data are identical 
    compatible = all(
        config_1[key][sub_key] == config_2[key][sub_key]
        for key in COMPATIBLE_DICT_PARAMETER
        for sub_key in COMPATIBLE_DICT_PARAMETER[key]
    )

    return identical, equal, compatible


# Data
def check_keys(data):
    necessary_keys = [
        "distances",
        "rounds",
        "noise_rates",
        "num_errors",
        "num_shots",
    ]
    if not all(key in necessary_keys for key in data.keys()):
        print(data.keys())
        raise Warning("Not all keys in data dict!") 
    pass


def write_data(data: dict, filepath: str):
    check_keys(data)    
    with h5py.File(filepath,"w") as file:
        for key in data:
            file.create_dataset(key, data=data[key]) 
    pass


def read_data(filepath):
    data = {}
    with h5py.File(filepath,"r") as file:
        for key in file:
            data[key] = np.array(file[key])
    """
    data should contain: 
        - distances
        - rounds
        - noise_rates
        - num_shots
        - num_errors 
    """
    return data 


def combine_data_sets(data_sets: list,):
    # not anymore checked if the configs agree!
    base_data_set = data_sets[0]
    if not all([ 
        all(base_data_set[key] == data_set[key])
        for key in DATA_SET_PARAMETER_KEYS
        for data_set in data_sets[1:]
        ]):
        raise UserWarning("combination of not equal datasets, not yet implemented!")

    for data_set in data_sets[1:]: 
        for key in DATA_SET_ADDITIVE_KEYS:
            base_data_set[key] += data_set[key] 
    return base_data_set 

# write results to folders: 
def folderpath_from_name(folder_name):
    base_path = PATH_TO_DATA_FOLDER 
    folder_path = os.path.join(base_path,folder_name)
    return folder_path


def write_folder(config, data, folder_name="", folder_path=""):
    if folder_name != "":
        folder_path = folderpath_from_name(folder_name)
    # create folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # config
    config_filepath = os.path.join(folder_path, "config.yaml")
    write_config(config, config_filepath)
    # data
    data_filepath = os.path.join(folder_path, "data.hdf5")
    write_data(data, data_filepath)
    pass


def read_folder(folder_name="",folder_path=""):
    if folder_name != "":
        folder_path = folderpath_from_name(folder_name)
    # config
    config_filepath = os.path.join(folder_path, "config.yaml")
    config = read_config(config_filepath)
    # data
    data_filepath = os.path.join(folder_path, "data.hdf5")
    data = read_data(data_filepath)
    return config, data

# smart functions (taking previous data and configs into account)
def smart_write_folder(config, data, folder_name="", folder_path="", unique_id: int = None):
    # enable saving of of multiple independet data rows to the same folder
    if folder_name != "":
        folder_path = folderpath_from_name(folder_name)

    if not os.path.exists(folder_path):
        # no previous folder 
        write_folder(config, data, folder_path=folder_path)
    else:
        # folder exist already
        pre_existing_config = read_config(os.path.join(folder_path, "config.yaml"))

        identical, equal, compatible = compare_configs(config, pre_existing_config)

        if identical or equal:
            data_string = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
            filename = DATA_BASE_NAME + "_" + data_string 
            if not unique_id is None :
                filename += "_" + str(unique_id) 
            filename += ".hdf5"
            filepath = os.path.join(folder_path, filename) 
            write_data(data, filepath)
            # either just write new data file or add data to existing data file
            pass
        elif compatible:
            # add data as new file!
            # TODO!
            print("Not yet implemented how to treat compatible configs!")
            pass
        else:
            # TODO!
            # should I warn the user?
            raise UserWarning("Folder already exists! and Configs are not compatible")
        pass


def smart_read_folder(folder_name="",folder_path=""):
    if folder_name != "":
        folder_path = folderpath_from_name(folder_name)
    # config
    config_filepath = os.path.join(folder_path, "config.yaml")
    config = read_config(config_filepath)
    # data (need to combine all present data sets)
    data_files = list(Path(folder_path).glob("*.hdf5"))
    data_sets = [read_data(data_file) for data_file in data_files]
    data = combine_data_sets(data_sets)
    return config, data


# other
def get_standard_config():
    # note lists should contain float or integers in yaml ... not good for noise rates
    return {
        "circuit": {
            "distances":    [3,],      # list of odd number >= 3
            "qec_rounds":       [1,],      # list of natural numbers
            "observable":   "Z",    # "Z" or "X" (maybe later also "B" for both)
            # not yet implemented
            "order":        "0p",   # order of ancilla CNOTs "0p", "p0" 
            "type": "steane",       # later also multiple rounds of surface code
            # redundant:
            "inital_state": "0",    # "0" or "p" (maybe later also "B" for bell state)
            "special_parameter": {},# open for future references
        },
        "noise_model": {
            "type": "circ",     # noise model: "circ", "bit_flip", "phase_flip", "basic" # this string defines the function that is gonna be used
            "noise_rates": [float(x) for x in np.logspace(-2.0,-0.8,dtype=float)],         # list of error rates
            "special_parameter": {},        # open for future references
        },
        "decoder": {
            "type": "ml",           # "ml" or "mwpm"
        },
        "sampling": {
            "num_shots": 1000,       # number of shots per configuration
        },
    }