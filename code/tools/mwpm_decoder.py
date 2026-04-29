import numpy as np
import pymatching

from tools.circuits import generate_simple_surface_code_circuit, generate_steane_circuit, generate_ft_surface_code_circuit
from tools.error_models import add_noise, construct_basic_noise_model 
from tools.error_propagation import uncorr_eff_noise
from tools.error_models import config_to_noise_model_func

# Steane code
def gen_mwpm_matcher(d, p, z_stab: bool = True, noise_model = "basic"): 
    # z_stab = false <=> decoder for x_stab 

    # treats X and Z errors independet!
    # select effective noise model (TODO: do I need to do that?! MWPM is cyclic in p and treats X and Z independetly (check!))
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
    # not needed right?!

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

# combined multi round steane code, shared info 
def gen_mwpm_matcher_shared_info(d, p, rounds, noise_model, observable):
    # TODO
    # here with the most basic approach of just generating a detector error model from the circuit
    circ = generate_steane_circuit(d,rounds=rounds,observable=observable,ft_stab=False)
    noise_model_func = config_to_noise_model_func({"noise_model": {"type": noise_model}}) # a bit cheaty...
    noisy_circ = add_noise(circ, noise_model_func(p))
    detector_error_model = noisy_circ.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    return matcher

# FT Surface code
def gen_mwpm_matcher_surface_code(d, p, noise_model, observable):
    # generate noise model
    circ = generate_ft_surface_code_circuit(
        d,
        rounds=1,
        observable=observable,
        ft_stab=False, # FT stabilizer are not included, treated seperatly
        )
    noise_model_func = config_to_noise_model_func({"noise_model": {"type": noise_model}}) # a bit cheaty...
    noisy_circ = add_noise(circ, noise_model_func(p))
    detector_error_model = noisy_circ.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    return matcher

def gen_mwpm_matcher_surface_code_with_FT(d, p, noise_model, observable):
    # generate noise model
    circ = generate_ft_surface_code_circuit(
        d,
        rounds=1,
        observable=observable,
        ft_stab=True, 
        )
    noise_model_func = config_to_noise_model_func({"noise_model": {"type": noise_model}}) # a bit cheaty...
    noisy_circ = add_noise(circ, noise_model_func(p))
    detector_error_model = noisy_circ.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    return matcher