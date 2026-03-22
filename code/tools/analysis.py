import numpy as np
# here i build the functions that take in the data object from files to do analysis on them
from tools.fssa import compute_critical_exponents, compute_error_bar 
from tools.graphics import plot_diff_noise_level, plot_fssa_results, overlay_different_slopes

# data anaylsis
def data_pre_processing(data: dict) -> dict:
    """
    This function:
        + calculates the log_error_rate
        + calculates the error on the log_error_rate 
    and adds those to the data dict 
    """
    n_d, n_r, n_p = len(data["distances"]), len(data["rounds"]), len(data["noise_rates"])
    num_shots = int(data["num_shots"])
    num_errors = data["num_errors"]

    log_error_rates = np.zeros((n_d,n_r,n_p))
    err_log_error_rates = np.zeros((n_d,n_r,n_p))
    for i_d in range(n_d):
        for i_r in range(n_r):
            for i_p in range(n_p):
                log_error_prob = num_errors[i_d,i_r,i_p]/num_shots
                log_error_rates[i_d,i_r,i_p] = log_error_prob 
                err_log_error_rates[i_d,i_r,i_p] = (log_error_prob*(1-log_error_prob)/num_shots)**(1/2)

    data["log_error_rates"] = log_error_rates 
    data["err_log_error_rates"] = err_log_error_rates
    return data

def determine_threshold(
        data: dict, 
        guess_nu: list, 
        guess_pth: list, 
        min_distance: int = 0, 
        min_noise_rate: float = 0,
        max_noise_rate: float = 1,
        select_rounds: list = None, 
        print_result: bool = True,
        ) -> dict:
    """
    This function determines the threshold of a given data set, for each rounds independently. 
    if <select_distances> and <select_rounds> are defined, only calculat for those.
    TODO: implement: select_rounds, select_distances
    TODO: implement Error calculation
    """
    if not "log_error_rates" in data.keys():
        print("WARNING: needed to calculate log_error_rates, for 'determine_threshold'")
        data = data_pre_processing(data)

    n_r = len(data["rounds"])
    ps = data["noise_rates"]
    distances = data["distances"]
    log_error_rates = data["log_error_rates"] 
    err_log_error_rates = data["err_log_error_rates"]

    # distance mask
    d_mask = distances >= min_distance
    # probability mask
    p_mask = (ps >= min_noise_rate) & (ps <= max_noise_rate)



    pth = np.zeros((n_r))
    nu = np.zeros((n_r)) 

    for i_r in range(n_r):
        result = compute_critical_exponents(
            xs_list=[ps[p_mask]]*len(distances[d_mask]),
            ys_list=log_error_rates[d_mask, i_r][:,p_mask],
            errs_list=err_log_error_rates[d_mask, i_r][:,p_mask],
            Ls=distances[d_mask],
            guess_xc=guess_pth[i_r],
            guess_nu=guess_nu[i_r],
        )
        if print_result:
            print()
            print(result)
        pth[i_r], inv_nu = result.x
        nu[i_r] = 1/inv_nu

        #TODO: Errorbars!
    
    data["p_threshold"] = pth
    data["nu_fit"] = nu

    return data

# Plotting
def data_plot_log_error_rates(data, rounds=None, min_distance: int = 0, min_noise_rate: float = 0, max_noise_rate: float = 1,  ):
    """
    This function plots all log errrot rates against the physical noise rate.
    if <distances> and <rounds> are defined, only plot those.  
    """
    if not "log_error_rates" in data.keys():
        print("WARNING: needed to calculate log_error_rates, for 'plot_log_error_rate'")
        data = data_pre_processing(data)

    n_d, n_r, n_p = len(data["distances"]), len(data["rounds"]), len(data["noise_rates"])
    distances = data["distances"]
    ps = data["noise_rates"]
    log_error_rates = data["log_error_rates"]
    err_log_error_rates = data["err_log_error_rates"]

    if "p_threshold" in data.keys():
        p_th = data["p_threshold"]
    else:
        p_th = [None] * n_r 

    # distance mask
    d_mask = distances >= min_distance
    # probability mask
    p_mask = (ps >= min_noise_rate) & (ps <= max_noise_rate)
    # for i_d in range(n_d):
    for i_r in range(n_r): 

        plot_diff_noise_level(
            log_error_rates = log_error_rates[d_mask, i_r][:,p_mask],
            y_errs = err_log_error_rates[d_mask, i_r][:,p_mask],
            distances = distances[d_mask],
            noise_set = ps[p_mask],
            p_th=p_th[i_r],
            )
    pass

def data_plot_fssa_results(data, min_distance: int = 0, min_noise_rate:float = 0, max_noise_rate:float = 1):
    if not "p_threshold" in data.keys():
        print("WARNING: needed to calculated p_threshold, for 'data_plot_fssa_results'")
        data = determine_threshold(data)

    n_r = len(data["rounds"])
    ps = data["noise_rates"]
    distances = data["distances"]
    log_error_rates = data["log_error_rates"] 
    err_log_error_rates = data["err_log_error_rates"]

    p_th = data["p_threshold"] 
    nu = data["nu_fit"] 

    # distance mask
    d_mask = distances >= min_distance
    # probability mask
    p_mask = (ps >= min_noise_rate) & (ps <= max_noise_rate)

    for i_r in range(n_r):
        plot_fssa_results(
            xs=[ps[p_mask]]*len(distances[d_mask]),
            ys=log_error_rates[d_mask, i_r][:,p_mask],
            yerrs=err_log_error_rates[d_mask, i_r][:,p_mask],
            distances=distances[d_mask],
            pc = p_th[i_r],
            nu = nu[i_r]
        )
    pass
