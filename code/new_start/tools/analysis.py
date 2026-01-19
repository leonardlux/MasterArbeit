import stim # type: ignore
import pymatching # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

from tools.error_models import add_noise

# TODO: update this function ?! decompose it into parts?!
def count_logical_errors_using_MWPM(circuit, num_shots: int, probability: bool = False) -> int:
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


# funciton to determin slope
def determine_slope(
        noise,
        log_prob,
        yerr=[],
        plot=False,
        plotpath="",):

    def linear(x,a,b):
        return a*x + b

    def clean_array(x,*args):
        """
        cleans out NaNs and infs from the x array,
        and applies the same mask to all args
        """
        mask = np.logical_and(np.logical_not(np.isnan(x)),np.isfinite(x))
        return x[mask], *[arg[mask] for arg in args] 


    # clean out NaNs and infite values, after log
    y, x= clean_array(np.log(log_prob), np.log(noise))

    # fit to curve on log scale
    popt, pcov = curve_fit(linear,x,y)
    exponent = popt[0]
    const = popt[1]
    
    if plot:
        plt.figure()
        if len(yerr)!=0:
            plt.errorbar(noise,log_prob,yerr=yerr,label="data points")
        else:
            plt.plot(noise,log_prob,label="data points")
        plt.plot(noise,np.exp(linear(np.log(noise),exponent,const)),label=f"fit: exp = {exponent:.4}")
        plt.xlabel('Physical error rate')
        plt.ylabel('Logical error rate')
        plt.legend(loc = 'upper left')
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
        if plotpath!= "":
            plt.savefig(plotpath)
    return exponent, const

def plot_diff_noise_level(
        circuits:list,
        noise_model_fct,
        noise_set = np.logspace(-2,-0.1),
        labels = None,
        num_shots = 10_000, 
        count_log_error_fct = count_logical_errors_using_MWPM,
        filename = "",
        plot_path = "/home/leo/Documents/MasterArbeit/code/images",
        fit_slopes = False,
        reference_lines = False,
    ):

    if not labels:
        legend = False
        labels = [""] * len(circuits)
    else:
        legend = True
    cm = 1/2.54 # to convert inches to cm
    plt.figure()
    plt.subplots(figsize=(20*cm,10*cm))
    plt.loglog()
    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per shot")

    log_error_probs = []
    y_errs = []
    for i,circuit in enumerate(circuits):
        log_error_prob = []
        for noise in noise_set:
            noisy_circuit = add_noise(
                circuit,
                noise_model_fct(noise),
                )
            log_error_prob.append(
                count_log_error_fct(noisy_circuit, num_shots, probability = True) 
                )
        
        log_error_prob = np.array(log_error_prob)
        y_err = (log_error_prob*(1-log_error_prob)/num_shots)**(1/2)

        log_error_probs.append(log_error_prob)
        y_errs.append(y_err)

        plt.errorbar(
            noise_set,
            log_error_prob,
            yerr=y_err,
            label=labels[i],
            )
        
        # TODO: add fitted reference lines!
        if fit_slopes:
            cutoff = int(len(noise_set)/2)
            exponent, const = determine_slope(noise_set[:cutoff],log_error_prob[:cutoff],plot=False)
            x = noise_set
            y = np.exp((np.log(noise_set)*exponent + const))
            plt.plot(
                x,
                y,
                label=f"fit {labels[i]}: exp = {exponent:.4}",
                linestyle="dotted",
                alpha=0.5,
                )

    if reference_lines:
        plt.plot(noise_set,noise_set**2,label="$p^2$")
        plt.plot(noise_set,noise_set,label="$p$",c="green")

    if legend:
        plt.legend()
    if filename != "":
        plt.savefig(plot_path +"/"+ filename +".pdf") # # TODO clean up using OS or similar
    plt.show()

    return 


    plt.loglog()

    plt.xlabel("physical error rate")
    plt.ylabel("logical error rate per shot")
    plt.legend()
    if filename != "":
        plt.savefig(plot_path + filename +".pdf")
    plt.show()

    return noise_set, log_error_probs, y_errs

