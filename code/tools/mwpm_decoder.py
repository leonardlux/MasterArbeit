import numpy as np
import pymatching

from tools.circuits import generate_simple_surface_code_circuit
from tools.error_models import add_noise, construct_basic_noise_model 
from tools.error_propagation import uncorr_eff_noise
from tools.syndrome import split_syndromes, split_and_xor_syndrome 

def gen_mwpm_matcher(d, p, z_stab: bool = True, noise_model = "basic"): 
    # z_stab = false <=> decoder for x_stab 

    # treats X and Z errors independet!
    # select effective noise model
    if noise_model == "circ":
        px, pz = uncorr_eff_noise(p)
    elif noise_model == "basic":
        px = p
        pz = p
    elif noise_model == "bit_flip":
        px = p
        pz = 0
    elif noise_model == "phase_flip":
        px = 0
        pz = p
    else:
        raise ValueError("unexpected noise model!")

    # determine relevant noise
    pr = px if z_stab else pz

    circ = generate_simple_surface_code_circuit(
        d,
        Z_stab=z_stab, 
        X_stab = not z_stab, 
        observable= ("Z" if z_stab else "X")
        )
    flip_noise = construct_basic_noise_model(
        pr, 
        X_errors=z_stab,
        Z_errors= not z_stab,
        )
    noisy_circ = add_noise(circ,flip_noise)
    detector_error_model = noisy_circ.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    return matcher
