import pymatching # type: ignore
import numpy as np # type: ignore
import numba 

from tools.error_models import add_noise
from tools.ml_decoder import decode_half_syndrome,  decode_half_syndrome_aron
from tools.mwpm_decoder import gen_mwpm_matcher
from tools.syndrome import split_and_xor_syndrome, reorder_syndromes 
from tools.pauli_frame_track import syndrome_to_pauli_flips 
from tools.error_propagation import uncorr_eff_noise

def sample_ciruit(circuit, num_shots):
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    return detection_events, observable_flips

def sample_circuit_format(circuit, d, observable, rounds, num_shots,):
    detection_events, observable_flips = sample_ciruit(circuit, num_shots) 

    ft_stab_z = True if observable == "Z" else False
    x_synds, z_synds, ft_synds = split_and_xor_syndrome(d, rounds, detection_events, ft_stab_z)

    if observable == "Z":
        rel_synd = z_synds
    elif observable == "X":
        rel_synd = x_synds
    else: 
        raise ValueError("observable value unexpected")
    return observable_flips, rel_synd, ft_synds

def count_logical_errors_using_MWPM_all_knowing_outdated(
        circuit, 
        num_shots: int, 
        probability: bool = False, 
        shortest_error: bool = False, 
        distance:int =0,
        error_rate: float = 0.,
        **kwargs, # just here to ignore the stuff other logical error counter need
        ) -> int:
    """
    This function realizes an MWPM Decoder that just builds an DEM from the circuit.
    Also known as all knowing MWMP Decoder, not comparable to ML Decoder (especially for circ noise or multi rounds) 
    """
    # Sample the circuit.
    detection_events, observable_flips = sample_ciruit(circuit,num_shots) 

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)
    
    if shortest_error:
        # Check the lowest weight error:
        error = detector_error_model.shortest_graphlike_error()
        print("The shortest possible error (found by stim) is formed by:")
        print(error)
        print(f"And it hast the lenght: {len(error)}")

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1

    if probability:
        return num_errors/num_shots
    return num_errors

def count_logical_errors_MWPM(
        circuit, 
        num_shots: int, 
        distance: int, 
        error_rate: float, 
        rounds: int,
        probability: bool = False,
        observable: str = "Z",
        noise_model: str = "circ",
        **kwargs, # just here to ignore the stuff other logical error counter need
    ):
    d = distance
    p = error_rate

    observable_flips, rel_synd, ft_synds = sample_circuit_format(circuit, d, observable, rounds, num_shots,)

    # Actual Decoding:
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

    mismatch = total_pred != observable_flips.flatten() 

    num_errors = np.sum(mismatch)  

    if probability:
        return num_errors/num_shots
    return num_errors 

# ML Decoding
def count_logical_errors_ML(
        circuit, 
        num_shots: int, 
        distance: int, 
        error_rate: float, 
        rounds: int,
        probability: bool = False,
        observable: str = "Z",
        noise_model: str = "circ",
        **kwargs, # just here to ignore the stuff other logical error counter need
    ):
    # select decoding implementation that gonna be used
    decode_half_syndrome_func = decode_half_syndrome_aron 
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

    observable_flips, rel_synd, ft_synds = sample_circuit_format(circuit, d, observable, rounds, num_shots,)

    # t_synd[round][shot][i_stab]
    rel_synd = reorder_syndromes(rel_synd)
    # rel_synd[shot][round][i_stab]

    # Actual Decoding: 
    predicitons = np.zeros((num_shots,rounds))
    pauli_repr_flips = np.zeros((num_shots,rounds))
    # all these calculation can be done in parralel!
    for i_shot in numba.prange(num_shots): 
        for i_round in numba.prange(rounds):
            predicitons[i_shot, i_round], pauli_repr_flips[i_shot,i_round] = decode_half_syndrome_func(
                d,
                p,
                rel_synd[i_shot,i_round],
            )
    multi_round_pred = np.sum(predicitons,axis=1)%2
    multi_round_pauli_flip = np.sum(pauli_repr_flips, axis=1)%2

    # FT prediciton
    # Old Way of ML FT prediction
    # ft_predictions = np.zeros((num_shots))
    # pauli_repr_flip_ft = np.zeros((num_shots))
    # for i_shot in numba.prange(num_shots):
    #     ft_predictions[i_shot], pauli_repr_flip_ft[i_shot] = decode_half_syndrome_func(
    #         d,
    #         p,
    #         ft_synds[i_shot] 
    #     )  
    # total_pred = (multi_round_pauli_flip + multi_round_pred + ft_predictions + pauli_repr_flip_ft)%2
    # MWPM Decoding -> better because symmetric (/cyclic) in error model choice for FT! = no error model choice needed
    z_stab = True if observable == "Z" else False
    matcher = gen_mwpm_matcher(d, p, z_stab, noise_model)
    ft_predictions = matcher.decode_batch(ft_synds).flatten()

    total_pred = (multi_round_pred + multi_round_pauli_flip + ft_predictions)%2
    total_pred = np.array(total_pred, dtype=bool) # convert to boolean values

    mismatch = total_pred != observable_flips.flatten() 
    num_errors = np.sum(mismatch)  

    if probability:
        return num_errors/num_shots
    return num_errors 

def generate_log_error_rates_diff_p(
        circuits:list,
        noise_model_fct,
        distances,
        rounds=1,
        noise_set = np.logspace(-2,-0.1),
        num_shots = 10_000, 
        noise_model = "circ",
        count_log_error_fct = count_logical_errors_using_MWPM_all_knowing_outdated,
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
            log_error_prob.append(
                count_log_error_fct(
                    noisy_circuit, 
                    num_shots= num_shots, 
                    rounds=rounds,
                    probability = True,
                    error_rate = noise,
                    distance = distances[i],
                    noise_model = noise_model,
                    ) 
                )
        
        log_error_prob = np.array(log_error_prob)
        y_err = (log_error_prob*(1-log_error_prob)/num_shots)**(1/2)

        log_error_rates.append(log_error_prob)
        y_errs.append(y_err)

    return log_error_rates, y_errs


