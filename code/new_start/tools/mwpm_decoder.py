import numpy as np
import pymatching

from tools.surface_code import generate_surface_code_circuit
from tools.error_models import add_noise, construct_basic_noise_model 
from tools.error_propagation import uncorr_eff_noise
from tools.helper import split_syndromes

def init_mwpm_decoder(d, p, z_stab: bool = True, noise_model = "circ_lvl"): 
    # z_stab = false <=> decoder for x_stab 

    # treats X and Z errors independet!
    # select effective noise model
    if noise_model == "circ_lvl":
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

    if z_stab:
        # only bit flip noise 
        X_errors = True
        Z_errors = False 
        pr = px 
    else:
        # only phase flip noise 
        X_errors = False 
        Z_errors = True 
        pr = pz

    circ = generate_surface_code_circuit(
        d,
        Z_stab=z_stab, 
        X_stab = not z_stab, 
        )
    flip_noise = construct_basic_noise_model(
        pr, 
        X_errors=X_errors,
        Z_errors=Z_errors,
        )
    n_circ = add_noise(circ,flip_noise)
    detector_error_model = n_circ.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    return matcher

def decode_mwpm_steane(d, p, syndromes, z_stab:bool = True, noise_model:str =  "circ_lvl"):
    """
    d: distance 
    p: error rate (assumes circuit lvl noise and uniform noise probability everywhere)
    syndromes: list of generated syndromes
    z_stab: True -> calculate prediciton log X (|0> state relevant)
            False-> calculate prediction log Z (|+> state relevant) <=> calcuates x_stab
    noise_model: defines which noise model the decoder assumes

    decoded using effective circuit and noise model!!
    treats X and Z noise independet 
    returns list of predictions for observable
    if both true returns list of list! 
    """
    x_stab_syndromes, z_stab_syndromes = split_syndromes(d, syndromes) 
    # Z-Stabilizer <=> X-Errors
    if z_stab:
        z_matcher = init_mwpm_decoder(d, p, z_stab=True, noise_model=noise_model)
        log_x_predictions = z_matcher.decode_batch(z_stab_syndromes)
        return log_x_predictions
    # X-Stabilizer <=> Z-Errors
    elif not z_stab:
        x_matcher = init_mwpm_decoder(d, p, z_stab=False, noise_model=noise_model)
        log_z_predictions = x_matcher.decode_batch(x_stab_syndromes)
        return log_z_predictions

# TODO build iterative multi round decoder! 
# TODO build for circuit level noise!