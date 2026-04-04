from tools.combined import generate_data_from_config
import numpy as np

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

x = generate_data_from_config(config)
print(x)