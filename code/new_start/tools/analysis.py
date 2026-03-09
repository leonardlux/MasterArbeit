import stim # type: ignore
import pymatching # type: ignore
import numpy as np # type: ignore

from tools.error_models import add_noise
from tools.ml_decoder import decode_half_syndrome,  decode_half_syndrome_aron
from tools.mwpm_decoder import decode_mwpm_steane, decode_mwpm_reps
from tools.helper import split_syndrome, split_syndromes_rounds, pauli_frame_track_syndromes
from tools.pauli_frame_track import syndrome_to_pauli_flips 

def sample_ciruit(circuit, num_shots):
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
    return detection_events, observable_flips

def gen_multi_rep_count_logical_error_MWPM(rounds:int = 1, init_state = "0",):
    if init_state=="0":
        z_stab = True
    def specific(
            circuit, 
            num_shots: int, 
            probability: bool = False, 
            distance:int =0,
            error_rate: float = 0.,
            **kwargs, # just here to ignore the stuff other logical error counter need
            ) -> int:
        """
        """
        d = distance 
        p = error_rate
        # Sample the circuit.
        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)


        # Run the decoder.
        predictions = decode_mwpm_reps(
            d, 
            p, 
            rounds,
            detection_events, 
            z_stab=True,
            ) 

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
    return specific 

def gen_error_model_count_logical_error_MWPM(noise_model,init_state = "0"):
    if init_state=="0":
        z_stab = True
    def specific(
            circuit, 
            num_shots: int, 
            probability: bool = False, 
            distance:int =0,
            error_rate: float = 0.,
            **kwargs, # just here to ignore the stuff other logical error counter need
            ) -> int:
        """
        """
        # Sample the circuit.
        sampler = circuit.compile_detector_sampler()
        detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

        # Run the decoder.
        predictions = decode_mwpm_steane(
            distance, 
            error_rate, 
            detection_events, 
            z_stab=True,
            noise_model=noise_model, # this is specified by parent function
            ) 

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
    return specific

def count_logical_errors_using_MWPM(
        circuit, 
        num_shots: int, 
        probability: bool = False, 
        shortest_error: bool = False, 
        distance:int =0,
        error_rate: float = 0.,
        **kwargs, # just here to ignore the stuff other logical error counter need
        ) -> int:
    """
    This funciton generates data from circuit and applies Minimum Weight Perfect Matching to decode the syndrome.
    It then return the amount of errors.

    It automaticly passes all detectors to the error model!
    
    :param circuit: Stim circuit obj that will be sampled 
    :param num_shots: Number of simulated processes 
    :type num_shots: int
    :param probability: If true returns the percentage of log mistakes instead of absoulte number 
    :type probabiliity: bool
    """
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

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

    # run ML decoding
    if False:
        d = distance 
        p = error_rate
        num_errors_ml = 0
        for obs, d_event in zip(observable_flips,detection_events): 
            x_stab_syndrome, z_stab_syndrome = split_syndrome(distance, d_event)
            pred_a = decode_half_syndrome_aron(
                d,
                p,
                z_stab_syndrome,
            )
            if obs[0] != pred_a:
                num_errors_ml += 1
        print(num_errors,num_errors_ml)
    if probability:
        return num_errors/num_shots
    return num_errors

def count_logical_errors_using_ML(
        circuit, 
        num_shots: int, 
        distance: int, 
        error_rate: float, 
        probability: bool = False,
        **kwargs, # just here to ignore the stuff other logical error counter need
    ) -> float:
    d = distance
    p = error_rate 
    
    detection_events, observable_flips = sample_ciruit(circuit, num_shots) 

    # init ML decoder (only X errors so far)
    num_errors = 0
    for obs, d_event in zip(observable_flips,detection_events): 
        x_stab_syndrome, z_stab_syndrome = split_syndrome(d, d_event)
        pred_log = decode_half_syndrome_aron(
            d,
            p,
            z_stab_syndrome,
        )
        log_x_f_p, log_z_f_p = syndrome_to_pauli_flips(d, d_event)

        pred_obs = pred_log ^ log_x_f_p

        if obs[0] != pred_obs:
            num_errors += 1
    if probability:
        return num_errors/num_shots
    return num_errors 

# function to determin slope
def generate_log_error_rates(
        circuits:list,
        noise_model_fct,
        distances,
        noise_set = np.logspace(-2,-0.1),
        num_shots = 10_000, 
        count_log_error_fct = count_logical_errors_using_MWPM,
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
                    num_shots, 
                    probability = True,
                    error_rate = noise,
                    distance = distances[i]
                    ) 
                )
        
        log_error_prob = np.array(log_error_prob)
        y_err = (log_error_prob*(1-log_error_prob)/num_shots)**(1/2)

        log_error_rates.append(log_error_prob)
        y_errs.append(y_err)

    return log_error_rates, y_errs


