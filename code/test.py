from tools.combined import generate_data_from_config
from tools.circuits import generate_ft_surface_code_circuit, generate_steane_circuit
from tools.error_models import add_noise, construct_circuit_noise_model
from tools.graphics import save_circuit_diagram

from tools.syndrome_prediction  import config_to_predict_func, sample_ciruit, calc_num_errors  

import numpy as np
import stim


config = {
        "circuit": {
            "distances":    [3,],      # list of odd number >= 3
            "qec_rounds":       [1,],      # list of natural numbers
            "observable":   "Z",    # "Z" or "X" (maybe later also "B" for both)
            # not yet implemented
            "order":        "0p",   # order of ancilla CNOTs "0p", "p0" 
            "type": "surface",       # later also multiple rounds of surface code
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
            "type": "mwpm",           # "ml" or "mwpm"
        },
        "sampling": {
            "num_shots": 1000,       # number of shots per configuration
        },
    }
d=3
rounds = 1
noise = 0.1
shots = 10

circ = generate_ft_surface_code_circuit(d,rounds,)
# circ = generate_steane_circuit(d,rounds)

n_circ = add_noise(circ, construct_circuit_noise_model(noise))

synd, obs = sample_ciruit(n_circ,10)
print(synd)
save_circuit_diagram(n_circ,"/home/fu494742/MasterArbeit/images/test.svg")
