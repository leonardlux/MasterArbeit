import stim # type: ignore
import pymatching # type: ignore
import numpy as np # type: ignore

def count_logical_errors_using_MWPM(circuit, num_shots: int) -> int:
    """
    This funciton generates data from circuit and applies Minimum Weight Perfect Matching to decode the syndrome.
    It then return the amount of errors.

    It automaticly passes all detectors to the error model!
    
    :param circuit: Description
    :param num_shots: Description
    :type num_shots: int
    :return: Description
    :rtype: int
    """
    # Sample the circuit.
    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    # Configure a decoder using the circuit.
    detector_error_model = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(detector_error_model)

    # Run the decoder.
    predictions = matcher.decode_batch(detection_events)

    # Count the mistakes.
    num_errors = 0
    for shot in range(num_shots):
        actual_for_shot = observable_flips[shot]
        predicted_for_shot = predictions[shot]
        if not np.array_equal(actual_for_shot, predicted_for_shot):
            num_errors += 1
    return num_errors

