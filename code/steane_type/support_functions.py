import numpy as np
import matplotlib.pyplot as plt

import stim
import pymatching

def count_logical_errors(circuit: stim.Circuit, num_shots: int) -> int:
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

# function that given a set of circuit factories, plots there logical error rate

def plot_factory_set(
        factory_set, 
        num_shots = 10_000,
        noise_set = np.logspace(-2,-0.1),
        filename = "",
        reference_lines=False,
        plot_path="/home/leo/Documents/MasterArbeit/code/steane_type/plots/"
    ):
    cm = 1/2.54  # centimeters in inches
    plt.figure()
    plt.subplots(figsize=(20*cm, 10*cm))
    for cur_circ_factory in factory_set:
        log_error_prob = []
        for noise in noise_set:
            circuit = cur_circ_factory(noise)
            num_errors_sampled = count_logical_errors(circuit, num_shots)
            log_error_prob.append(num_errors_sampled / num_shots)
        log_error_prob = np.array(log_error_prob)
        plt.errorbar(
            noise_set,
            log_error_prob,
            yerr=(log_error_prob*(1-log_error_prob)/num_shots)**(1/2),
            label=cur_circ_factory(0,name=True),
            )
    plt.loglog()
    if reference_lines:
        plt.plot(noise_set,noise_set**2,label="$p^2$")
        plt.plot(noise_set,noise_set,label="$p$",c="green")
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per shot")
    plt.legend()
    if filename != "":
        plt.savefig(plot_path + filename +".pdf")
    plt.show()