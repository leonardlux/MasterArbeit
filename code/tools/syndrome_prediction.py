import numpy as np # type: ignore
import numba 

from tools.error_models import add_noise
from tools.ml_decoder import decode_half_syndrome, decode_half_syndrome_log,  decode_half_syndrome_aron, maybe_jit
from tools.mwpm_decoder import gen_mwpm_matcher, gen_mwpm_matcher_surface_code, gen_mwpm_matcher_surface_code_with_FT, pred_pauli_frame_track_repeated_surface_code
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

# MWPM prediction for Steane Type error correction 
# Assumes syndrome are extracted faultless and then corrects them based on the DEM of a code capacity surface code with standard syndrome extraction
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
    predictions = np.zeros((rounds, num_shots))
    z_stab = True if observable == "Z" else False
    matcher = gen_mwpm_matcher(d, p, z_stab, noise_model)
    for i_round in range(rounds):
        predictions[i_round] = matcher.decode_batch(rel_synd[i_round]).flatten()
        # .flatten() is needed because we always assume that only one observable is measured (in ML, and I wanted to adapt to this problem)
    # combine rounds together
    multi_round_pred =np.sum(predictions,axis=0)%2
    # FT Decoding 
    # we use the same matcher because the exact error value is irrelevant for MWPM (cyclic/symmetric in it)
    ft_predictions = matcher.decode_batch(ft_synds).flatten()

    total_pred = (multi_round_pred + ft_predictions)%2
    total_pred = np.array(total_pred, dtype=bool)

    fault_flags = np.zeros(detection_events.shape[0],dtype=np.bool) 
    return total_pred, fault_flags

"""
MWPM for REPEATED (standard) SYNDROME READOUT
"""
# TODO: DOES NOT WORK! Last measurement error is not fixed. One would need sliding window approach or similiar -> for upper bound, use full info version! 
# The equations are at the moment not fully correct, here I use s_i = sum D_i, which is only true if the last measurement error is included 
def predict_MWPM_rep_surface_code(
        detection_events, 
        distance: int, 
        error_rate: float, 
        rounds: int,
        observable: str = "Z",
        noise_model: str = "circ",
    ):
    d = distance
    p = error_rate
    z_stab = True if observable == "Z" else False

    # proper disconnected implementation
    qec_round_syndromes, ft_synds = preprocess_surface_code_syndromes(
        d= d,
        rounds=rounds,
        syndromes=detection_events,
    )
    num_shots, rounds, detectors_per_round = qec_round_syndromes.shape

    if detectors_per_round != 2*d*d*(d-1):
        # this is not happening, just here to test for possible errors
        print("Something wrong, unexpected amount of detectors for repeated syndrome readout")

    # Actual Decoding:
    matcher = gen_mwpm_matcher_surface_code(d, p, noise_model, observable=observable)
    predictions = np.zeros((rounds, num_shots))
    for i_shot in range(num_shots):
        num_detectors_simple_surface_code = 2*d*(d-1)
        pauli_tracking_syndrome = np.zeros(num_detectors_simple_surface_code,dtype=bool)
        for i_round in range(rounds):
            predictions[i_round,i_shot], pauli_tracking_syndrome = pred_pauli_frame_track_repeated_surface_code(
                d=d,
                matcher=matcher,
                syndrome=qec_round_syndromes[i_shot,i_round,:],
                pauli_tracking_syndrome=pauli_tracking_syndrome,
            )  
        # Pauli frame tracking applied to FT check -> syndrome of residual error
        # if z_stab: 
        #     # last d*(d-1) detectors are z stabilizers
        #     ft_synds[i_round] ^= pauli_tracking_syndrome[-d*(d-1):]
        # else:
        #     # first d*(d-1) detectors are x stabilizers
        #     ft_synds[i_round] ^= pauli_tracking_syndrome[:d*(d-1)]

    # combine rounds together
    multi_round_pred = np.sum(predictions,axis=0)%2

    # FT Decoding (Same as usual)
    matcher = gen_mwpm_matcher(d, p, z_stab, noise_model="basic")
    ft_predictions = matcher.decode_batch(ft_synds).flatten()

    total_pred = (multi_round_pred + ft_predictions)%2
    total_pred = np.array(total_pred, dtype=bool)

    return total_pred 

# decoding of repeated syndrome extraction with FULL INFO (decoding including FT)
def predict_MWPM_rep_surface_code_full_info(
        detection_events, 
        distance: int, 
        error_rate: float, 
        rounds: int,
        observable: str = "Z",
        noise_model: str = "circ",
    ):
    d = distance
    p = error_rate

    # complete circuit and FT measurement is feed into decoder
    matcher = gen_mwpm_matcher_surface_code_with_FT(d, p, noise_model, observable=observable, rounds=rounds)
    total_pred = matcher.decode_batch(detection_events).flatten()

    # no faulty decoding -> trivial array
    fault_flags = np.zeros(detection_events.shape[0],dtype=np.bool) 
    return total_pred, fault_flags


"""
ML Decoding (for Steane code/assuming no measurement errors)
"""

@maybe_jit
def decoding(d,p,observable,rel_synd, decode_half_syndrome_func, dtype):    
    num_shots, rounds, _ = rel_synd.shape
    matrix_shape = (num_shots,rounds)
    predictions = np.zeros(matrix_shape)
    pauli_repr_flips = np.zeros(matrix_shape)
    faults = np.zeros(matrix_shape)
    for i_round in numba.prange(rounds):
        for i_shot in range(num_shots): 
            predictions[i_shot, i_round], pauli_repr_flips[i_shot,i_round], faults[i_shot,i_round] = decode_half_syndrome_func(
                d,
                p,
                rel_synd[i_shot,i_round],
                stab_type=observable, # the observable determines which stabilizers we need to decode
                dtype=dtype,
            )
    multi_round_pred = np.sum(predictions,axis=1)%2
    multi_round_pauli_flip = np.sum(pauli_repr_flips, axis=1)%2
    # fault_flags = faults.any(axis=1) # not supported in numba, therefore done manually
    fault_flags = np.zeros(faults.shape[0],dtype=np.bool)
    for i in range(faults.shape[0]): # shots
        flag = False
        for j in range(faults.shape[1]): # rounds
            if faults[i, j]:
                flag = True
                break
        fault_flags[i] = flag

    return multi_round_pred, multi_round_pauli_flip, fault_flags

def factory_predict_func_ML(
        decoding_func = decode_half_syndrome_log,
        dtype = np.float64, # standard from numpy
        ft_mwpm = True,
):
    def predict_ML(
            detection_events, 
            distance: int, 
            error_rate: float, 
            rounds: int,
            observable: str = "Z",
            noise_model: str = "circ",
        ):
        # select decoding implementation that gonna be used
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
        multi_round_pred, multi_round_pauli_flip, fault_flags = decoding(d, p, observable, rel_synd, decoding_func, dtype)

        # FT Decoding 
        if ft_mwpm:
            # MWPM
            z_stab = True if observable == "Z" else False
            matcher = gen_mwpm_matcher(d, p, z_stab, noise_model)
            ft_predictions = matcher.decode_batch(ft_synds).flatten()
        else:
            # ML
            num_shots, _, _ = rel_synd.shape
            predictions_FT = np.zeros(num_shots)
            pauli_repr_flips_FT= np.zeros(num_shots)
            for i_shot in range(num_shots): 
                predictions_FT[i_shot], pauli_repr_flips_FT[i_shot], faults_FT = decoding_func(
                    d,
                    p,
                    ft_synds[i_shot],
                    stab_type=observable, # the observable determines which stabilizers we need to decode
                    dtype=dtype,
                )
                if faults_FT:
                    fault_flags[i_shot] = faults_FT
            ft_predictions = (predictions_FT + pauli_repr_flips_FT)%2

            #modify fault flags
            
        total_pred = (multi_round_pred + multi_round_pauli_flip + ft_predictions)%2
        total_pred = np.array(total_pred, dtype=bool) # convert to boolean values
        return total_pred, fault_flags
    return predict_ML


def config_to_predict_func(config):
    circuit_type = config["circuit"]["type"]
    value = config["decoder"]["type"]
    if circuit_type == "steane":
        if value == "ml":
            if  "special_parameter" in config["decoder"] and "ft_mwpm" in config["decoder"]["special_parameter"]:
                return factory_predict_func_ML(
                    ft_mwpm = config["decoder"]["special_parameter"]["ft_mwpm"],
                )
            else:
                return factory_predict_func_ML()  # basic configuration
        elif value == "mwpm":
            return predict_MWPM
        elif value == "ml_test":
            # test cases for ML decoding (check num. precission) 
            # Data Type options
            data_type = config["decoder"]["special_parameter"]["data_type"]
            if data_type == 16:
                dtype = np.float16 # not completly implemented
            elif data_type == 32:
                dtype = np.float32
            elif data_type == 64:
                dtype = np.float64
            elif data_type == 128:
                dtype = np.float128 # not completly implemented
            # Log or not (and aron or not?)
            decode_func_str = config["decoder"]["special_parameter"]["decode_str"]
            if decode_func_str == "log":
                decode_func = decode_half_syndrome_log
            elif decode_func_str == "basic":
                decode_func = decode_half_syndrome
            elif decode_func_str == "aron":
                decode_func = decode_half_syndrome_aron
            else:
                raise ValueError()
            return factory_predict_func_ML(
                decoding_func=decode_func,
                dtype=dtype,
            )

        else:
            raise ValueError("Unknwon Decoder Type")

    elif circuit_type == "surface":
        if value == "mwpm":
            print("This Function is not working!")
            return predict_MWPM_rep_surface_code
        elif value == "mwpm_full_info":
            return predict_MWPM_rep_surface_code_full_info
        else:
            raise ValueError("Unknown Decoder Type")
    else:
        raise ValueError("Unkown Circuit Type")