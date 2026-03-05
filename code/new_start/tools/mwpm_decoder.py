import numpy as np
import pymatching

from tools.surface_code import generate_surface_code_circuit
from tools.error_models import add_noise, construct_basic_noise_model 
from tools.error_propagation import uncorr_eff_noise
from tools.helper import split_syndromes, split_and_pauli_frame_track 

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
    x_stab_syndromes, z_stab_syndromes, ft_syndromes = split_syndromes(d, syndromes) 

    # z_stab==True: Z-Stabilizer <=> X-Errors
    if z_stab: 
        stab_syndromes = z_stab_syndromes
    # z_stab==False: X-Stabilizer <=> Z-Errors
    elif not z_stab:
        stab_syndromes = x_stab_syndromes

    matcher = init_mwpm_decoder(d, p, z_stab, noise_model=noise_model)
    log_predictions = matcher.decode_batch(stab_syndromes)

    if noise_model == "circ_lvl":
        # assuming same error probability for ft error (TODO correct this!)
        ft_syndromes = np.logical_xor(ft_syndromes,stab_syndromes) # pauli frame tracking!!! idea
        ft_matcher = init_mwpm_decoder(d,p) 
        ft_predicitons = ft_matcher.decode_batch(ft_syndromes)
        # xor works!
        log_predictions = log_predictions^ft_predicitons

    return log_predictions

def decode_mwpm_reps(d, p, reps, syndromes, z_stab:bool = True,):
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
    # pauli frame tracked syndromes
    px_synd, pz_synd, pft_synd = split_and_pauli_frame_track(d, reps, syndromes, z_stab)

    # z_stab==True: Z-Stabilizer <=> X-Errors
    if z_stab: 
        stab_syndromes = pz_synd 
    # z_stab==False: X-Stabilizer <=> Z-Errors
    elif not z_stab:
        stab_syndromes = px_synd 

    shots = len(stab_syndromes[0])
    log_predictions = np.zeros((reps+1,shots,1)) # reps +1 for ft-check

    # Steane syndromes
    matcher = init_mwpm_decoder(d, p, z_stab, noise_model="circ_lvl")
    for rep in range(reps):
        log_predictions[rep] = matcher.decode_batch(stab_syndromes[rep])
    # FT syndrome
    ft_matcher = init_mwpm_decoder(d, p, z_stab, noise_model="circ_lvl")
    log_predictions[reps] = ft_matcher.decode_batch(pft_synd)

    final_predictions = np.zeros((shots))
    for i in range(len(log_predictions)):
        final_predictions = np.logical_xor(final_predictions,log_predictions[i])

    return final_predictions
# TODO correct error probability for FT syndrome (can I calc that?) 
# TODO build iterative multi round decoder! 