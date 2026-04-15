
# PATH_TO_CONFIG_FOLDER = "/home/leo/Documents/MasterArbeit/code/configs"
# PATH_TO_DATA_FOLDER = "/home/leo/Documents/MasterArbeit/code/data" 

PATH_TO_CONFIG_FOLDER = "/home/fu494742/MasterArbeit/code/configs"
PATH_TO_DATA_FOLDER =  "/home/fu494742/MasterArbeit/code/data"
PATH_TO_IMAGE_FOLDER = "/home/fu494742/MasterArbeit/images"

DATA_BASE_NAME = "data"

# defines which dict parameters needs to be identical for two configs to add to the same data set
# other parameters: num_shots, noise_rates, distances, qec_rounds are all given and sourced from data file
# given parameters are examples
COMPATIBLE_DICT_PARAMETER = {
        "circuit": {
            "observable":   "Z",    
            "order":        "0p",   
            "type":         "steane",       
            "inital_state": "0",    
        },
        "noise_model": {
            "type": "circ",  
        },
        "decoder": {
            "type": "ml",
        },
    }

DATA_SET_ADDITIVE_KEYS = [
    "num_shots",
    "num_errors",
]

DATA_SET_PARAMETER_KEYS = [
    "distances",
    "rounds",
    "noise_rates",
]

# Nearly Identical Configs
EQUAL_CONFIGS_EXCLUDE_KEY = "sampling"
