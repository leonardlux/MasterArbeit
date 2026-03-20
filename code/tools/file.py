import h5py
import yaml
import os 
import numpy as np

# Configs
def write_config(configuration: dict, filepath: str):
    with open(filepath,"w") as file:
        yaml.dump(configuration, file)
    pass

def read_config(filepath: str):
    with open(filepath, "r") as file:
        configuration = yaml.safe_load(file)
    return configuration 


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
    return data 

# write results to folders: 
def folderpath_from_name(folder_name):
    base_path = "/home/leo/Documents/MasterArbeit/code/data"
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


# other
def get_basic_config():
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
            "noise_model_name": "circ",     # noise model: "circ", "bit_flip", "phase_flip", "basic" # this string defines the function that is gonna be used
            "noise_rates":   [0.1],         # list of error rates
            "special_parameter": {},        # open for future references
        },
        "decoder": {
            "type": "ml",           # "ml" or "mwpm"
        },
        "sampling": {
            "num_shots": 1000,       # number of shots per configuration
        },
    }

