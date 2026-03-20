import numpy as np

from tools.circuits             import config_to_circ_func
from tools.error_models         import config_to_noise_model_func, add_noise 
from tools.syndrome_prediction  import config_to_predict_func, sample_ciruit, calc_num_errors  
from tools.file                 import read_config, write_folder


def generate_data_from_config(config: dict):
    # baseline config 
    circ_func = config_to_circ_func(config)
    observable = config["circuit"]["observable"]
    noise_model_func = config_to_noise_model_func(config)
    noise_model_type = config["noise_model"]["type"]
    predict_func = config_to_predict_func(config)
    num_shots = config["sampling"]["num_shots"]

    # iterable config 
    ds = config["circuit"]["distances"]
    n_rounds = config["circuit"]["qec_rounds"]
    ps = config["noise_model"]["noise_rates"]

    num_errors = np.zeros((len(ds),len(n_rounds),len(ps)))
    for i_d, d in enumerate(ds):
        for i_r, rounds in enumerate(n_rounds):
            # generate circ layout
            circ_d_r = circ_func(
                distance =  d,
                rounds =    rounds,
                observable =observable,
                )
            for i_p, p in enumerate(ps):
                # add noise
                circ_d_r_p = add_noise(
                    circuit =       circ_d_r,  
                    noise_model =   noise_model_func(p),
                    )
                # sample circuit 
                detection_events, observable_flips = sample_ciruit(circ_d_r_p, num_shots) 
                # decode syndromes
                predictions = predict_func(
                    detection_events,
                    distance=d,
                    error_rate=p,
                    rounds=rounds,
                    observable=observable,
                    noise_model=noise_model_type,
                )
                # compare predicitons and obs
                num_errors[i_d,i_r,i_p] = calc_num_errors(predictions,observable_flips)

    data = {
        "distances": ds,
        "rounds": n_rounds,
        "noise_rates": ps,
        "num_errors": num_errors,
        "num_shots": num_shots,
    }
    return data 

def generate_new_data_from_config_file(config_filepath, output_folder_name):
    config = read_config(config_filepath)
    data = generate_data_from_config(config)
    write_folder(config, data, folder_name=output_folder_name)

