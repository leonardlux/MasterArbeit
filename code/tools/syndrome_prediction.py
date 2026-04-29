import pymatching # type: ignore
import numpy as np # type: ignore
import numba 

from tools.error_models import add_noise
from tools.ml_decoder import decode_half_syndrome,  decode_half_syndrome_aron
from tools.mwpm_decoder import gen_mwpm_matcher, gen_mwpm_matcher_surface_code, gen_mwpm_matcher_surface_code_with_FT
from tools.syndrome import split_and_xor_syndrome, reorder_syndromes, preprocess_surface_code_syndromes
from tools.error_propagation import uncorr_eff_noise

# General stuff
def sample_ciruit(circuit, num_shots):
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    return detection_events, observable_flips

def format_syndromes(d, observable, rounds, detection_events):

    ft_stab_z = True if observable == "Z" else False
    x_synds, z_synds, ft_synds = split_and_xor_syndrome(d, rounds, detection_events, ft_stab_z)

    if observable == "Z":
        rel_synd = z_synds
    elif observable == "X":
        rel_synd = x_synds
    else: 
        raise ValueError("observable value unexpected")
    return rel_synd, ft_synds

def calc_num_errors(pred,obs):    
    mismatch = pred != obs.flatten() 
    num_errors = np.sum(mismatch)  
    return num_errors

def predict_MWPM(
        detection_events, 
        distance: int, 
        error_rate: float, 
        rounds: int,
        observable: str = "Z",
        noise_model: str = "circ",
    ):
    d = distance
    p = error_rate
    rel_synd, ft_synds = format_syndromes(d, observable, rounds, detection_events)

    # Actual Decoding:
    rounds, num_shots, _ = rel_synd.shape
    predicitons = np.zeros((rounds, num_shots))
    z_stab = True if observable == "Z" else False
    matcher = gen_mwpm_matcher(d, p, z_stab, noise_model)
    for i_round in range(rounds):
        predicitons[i_round] = matcher.decode_batch(rel_synd[i_round]).flatten()
        # .flatten() is needed because we always assume that only one observable is measured (in ML, and I wanted to adapt to this problem)
    # combine rounds together
    multi_round_pred =np.sum(predicitons,axis=0)%2
    # FT Decoding 
    # we use the same matcher because the exact error value is irrelvant for MWPM (cyclic/symmetric in it)
    ft_predictions = matcher.decode_batch(ft_synds).flatten()

    total_pred = (multi_round_pred + ft_predictions)%2
    total_pred = np.array(total_pred, dtype=bool)

    return total_pred


# Surface code:
def predict_MWPM_surface_code(
        detection_events, 
        distance: int, 
        error_rate: float, 
        rounds: int,
        observable: str = "Z",
        noise_model: str = "circ",
    ):
    d = distance
    p = error_rate
    # shortcut of complete circuit 1 round with FT
    matcher = gen_mwpm_matcher_surface_code_with_FT(d, p, noise_model, observable=observable)
    total_pred_whole = matcher.decode_batch(detection_events).flatten()

    # propper disconnected implementation
    qec_round_syndromes, ft_synds = preprocess_surface_code_syndromes(
        d= d,
        rounds=rounds,
        syndromes=detection_events,
    )

    # Actual Decoding:
    num_shots, rounds, _ = qec_round_syndromes.shape
    predicitons = np.zeros((rounds, num_shots))
    matcher = gen_mwpm_matcher_surface_code(d, p, noise_model, observable=observable)
    for i_round in range(rounds):
        predicitons[i_round] = matcher.decode_batch(qec_round_syndromes[:,i_round,:]).flatten()
        # .flatten() is needed because we always assume that only one observable is measured (in ML, and I wanted to adapt to this problem)
    # combine rounds together
    multi_round_pred = np.sum(predicitons,axis=0)%2

    # FT Decoding (Same as usual)
    z_stab = True if observable == "Z" else False
    matcher = gen_mwpm_matcher(d, p, z_stab, noise_model="basic")
    ft_predictions = matcher.decode_batch(ft_synds).flatten()

    total_pred = (multi_round_pred + ft_predictions)%2
    total_pred = np.array(total_pred, dtype=bool)

    return total_pred_whole


# ML Decoding
# TODO: is parralism acutally an improvement? not if I just want to use a single core!
@numba.njit()#parallel=True)
def fast_decoding(d,p,observable,rel_synd):    
    num_shots, rounds, _ = rel_synd.shape
    predicitons = np.zeros((num_shots,rounds))
    pauli_repr_flips = np.zeros((num_shots,rounds))
    for i_round in numba.prange(rounds):
        for i_shot in range(num_shots): 
            predicitons[i_shot, i_round], pauli_repr_flips[i_shot,i_round] = decode_half_syndrome_aron(
                d,
                p,
                rel_synd[i_shot,i_round],
                stab_type=observable, # the observable determines which stabilizers we need to decode
            )
    multi_round_pred = np.sum(predicitons,axis=1)%2
    multi_round_pauli_flip = np.sum(pauli_repr_flips, axis=1)%2
    return multi_round_pred, multi_round_pauli_flip

# TODO Jit this one!
def slow_decoding(d,p,observable,rel_synd):    
    num_shots, rounds, _ = rel_synd.shape
    predicitons = np.zeros((num_shots,rounds))
    pauli_repr_flips = np.zeros((num_shots,rounds))
    # all these calculation can be done in parralel!
    for i_shot in numba.prange(num_shots): 
        for i_round in numba.prange(rounds):
            predicitons[i_shot, i_round], pauli_repr_flips[i_shot,i_round] = decode_half_syndrome(
                d,
                p,
                rel_synd[i_shot,i_round],
                stab_type=observable, # the observable determines which stabilizers we need to decode
            )
    multi_round_pred = np.sum(predicitons,axis=1)%2
    multi_round_pauli_flip = np.sum(pauli_repr_flips, axis=1)%2
    return multi_round_pred, multi_round_pauli_flip

def predict_ML(
        detection_events, 
        distance: int, 
        error_rate: float, 
        rounds: int,
        observable: str = "Z",
        noise_model: str = "circ",
    ):
    # select decoding implementation that gonna be used
    decoding_func = fast_decoding 
    d = distance
    # Adapt noise to given noise model
    if noise_model == "circ":
        px, pz = uncorr_eff_noise(error_rate)
        if observable == "Z":
            p = px
        elif observable == "X":
            p = pz
        else:
            raise ValueError("Unexpected observable value")
    elif noise_model in ["bit_flip", "basic", "phase_flip"]:
        p = error_rate
    else:
        raise ValueError("Unexpected noise_model value")

    rel_synd, ft_synds = format_syndromes(d, observable, rounds, detection_events)

    # t_synd[round][shot][i_stab]
    rel_synd = reorder_syndromes(rel_synd)
    # rel_synd[shot][round][i_stab]

    # Actual Decoding: 
    multi_round_pred, multi_round_pauli_flip = decoding_func(d,p,observable,rel_synd)

    # FT Decoding (MWPM)
    z_stab = True if observable == "Z" else False
    matcher = gen_mwpm_matcher(d, p, z_stab, noise_model)
    ft_predictions = matcher.decode_batch(ft_synds).flatten()

    total_pred = (multi_round_pred + multi_round_pauli_flip + ft_predictions)%2
    total_pred = np.array(total_pred, dtype=bool) # convert to boolean values
    return total_pred 



# NOT NEEDED ?!
def generate_log_error_rates_diff_p(
        circuits:list,
        noise_model_fct,
        distances,
        rounds=1,
        observable:int = "Z",
        noise_set = np.logspace(-2,-0.1),
        num_shots = 10_000, 
        noise_model = "circ",
        count_log_error_fct = predict_ML,
    ):
    log_error_rates = []
    y_errs = []
    for i,circuit in enumerate(circuits):
        log_error_prob = []
        for noise in noise_set:
            noisy_circuit = add_noise(
                circuit,
                noise_model_fct(noise),
                )
            detection_events, observable_flips = sample_ciruit(noisy_circuit, num_shots)
            predictions = count_log_error_fct(
                    detection_events, 
                    rounds=rounds,
                    error_rate = noise,
                    distance = distances[i],
                    noise_model = noise_model,
                    observable = observable,
                    ) 
            num_errors = calc_num_errors(predictions, observable_flips)
            log_error_prob.append(
                num_errors/num_shots
               )
        
        log_error_prob = np.array(log_error_prob)
        y_err = (log_error_prob*(1-log_error_prob)/num_shots)**(1/2)

        log_error_rates.append(log_error_prob)
        y_errs.append(y_err)
    return log_error_rates, y_errs

def config_to_predict_func(config):
    circuit_type = config["circuit"]["type"]
    value = config["decoder"]["type"]
    if circuit_type == "steane":
        if value == "ml":
            return predict_ML 
        elif value == "mwpm":
            return predict_MWPM
        else:
            raise ValueError()
    elif circuit_type == "surface":
        if value == "mwpm":
            return predict_MWPM_surface_code
        else:
            raise ValueError()
    else:
        raise ValueError()