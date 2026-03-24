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
        min_distance: list = None, 
        min_noise_rate: list = None,
        max_noise_rate: list = None,
        select_rounds: list = None, 
        print_result: bool = True,
        ) -> dict:
    """
    This function determines the threshold of a given data set, for each rounds independently. 
    if select_rounds != None only calculate for the given number of qec rounds.
    """
    if not "log_error_rates" in data.keys():
        print("WARNING: needed to calculate log_error_rates, for 'determine_threshold'")
        data = data_pre_processing(data)

    rounds = data["rounds"]
    n_r = len(rounds)
    ps = data["noise_rates"]
    distances = data["distances"]
    log_error_rates = data["log_error_rates"] 
    err_log_error_rates = data["err_log_error_rates"]

    # initalize proper standard values
    if min_distance == None:
        min_distance = [0] * len(rounds)
    if min_noise_rate == None:
        min_noise_rate = [0] * len(rounds)
    if max_noise_rate == None:
        max_noise_rate = [1] *len(rounds)

    pth = np.zeros((n_r))
    err_pth = np.zeros((n_r))
    nu = np.zeros((n_r))
    err_nu = np.zeros((n_r))

    for i_r in range(n_r):
        if select_rounds != None and not rounds[i_r] in select_rounds:
            continue # skip not selected rounds if 

        # distance mask
        d_mask = distances >= min_distance[i_r]
        # probability mask
        p_mask = (ps >= min_noise_rate[i_r]) & (ps <= max_noise_rate[i_r])

        xs_list=[ps[p_mask]]*len(distances[d_mask])
        ys_list=log_error_rates[d_mask, i_r][:,p_mask]
        errs_list=err_log_error_rates[d_mask, i_r][:,p_mask]
        Ls = distances[d_mask]

        result = compute_critical_exponents(
            xs_list=xs_list,
            ys_list=ys_list,
            errs_list=errs_list,
            Ls=Ls,
            guess_xc=guess_pth[i_r],
            guess_nu=guess_nu[i_r],
        )
        if print_result:
            print()
            print(result)

        pth[i_r], inv_nu = result.x
        nu[i_r] = 1/inv_nu

        err_pth[i_r], inv_nu_err = compute_error_bar(
            xs_list=xs_list,
            ys_list=ys_list,
            errs_list=errs_list,
            Ls=Ls,
            guess_xc=guess_pth[i_r],
            guess_nu=guess_nu[i_r],
        )
        err_nu[i_r] = inv_nu_err/nu[i_r]**2
    
    print(pth)
    print(err_pth)
    
    data["p_th"] = pth
    data["err_p_th"] = err_pth 

    data["nu_fit"] = nu
    data["err_nu_fit"] = err_nu

    return data

# Plotting
def data_plot_log_error_rates(data, select_rounds: list = None, min_distance: list = None, min_noise_rate: list  = None, max_noise_rate: list = None,  ):
    """
    This function plots all log errrot rates against the physical noise rate.
    if select_rounds != None only calculate for the given number of qec rounds.
    """
    if not "log_error_rates" in data.keys():
        raise ValueError("WARNING: needed to calculate log_error_rates, for 'plot_log_error_rate'")

    # read data from data dict 
    rounds = data["rounds"]
    distances = data["distances"]
    ps = data["noise_rates"]
    log_error_rates = data["log_error_rates"]
    err_log_error_rates = data["err_log_error_rates"]

    # initalize proper standard values
    if min_distance == None:
        min_distance = [0] * len(rounds)
    if min_noise_rate == None:
        min_noise_rate = [0] * len(rounds)
    if max_noise_rate == None:
        max_noise_rate = [1] *len(rounds)

    if "p_th" in data.keys():
        p_th = data["p_th"]
    else:
        p_th = [None] * len(rounds) 

    # for i_d in range(n_d):
    for i_r in range(len(rounds)): 
        if select_rounds != None and not rounds[i_r] in select_rounds:
            continue # skip not selected rounds if 

        # distance mask
        d_mask = distances >= min_distance[i_r]
        # probability mask
        p_mask = (ps >= min_noise_rate[i_r]) & (ps <= max_noise_rate[i_r])

        plot_diff_noise_level(
            log_error_rates = log_error_rates[d_mask, i_r][:,p_mask],
            y_errs = err_log_error_rates[d_mask, i_r][:,p_mask],
            distances = distances[d_mask],
            noise_set = ps[p_mask],
            p_th=p_th[i_r],
            )
    pass

def data_plot_fssa_results(data, min_distance: list = None, min_noise_rate: list = None, max_noise_rate:list = None, select_rounds: list = None):
    if not "p_th" in data.keys():
        print("WARNING: needed to calculated p_threshold, for 'data_plot_fssa_results'")
        data = determine_threshold(data)

    rounds = data["rounds"]
    ps = data["noise_rates"]
    distances = data["distances"]
    log_error_rates = data["log_error_rates"] 
    err_log_error_rates = data["err_log_error_rates"]

    # initalize proper standard values
    if min_distance == None:
        min_distance = [0] * len(rounds)
    if min_noise_rate == None:
        min_noise_rate = [0] * len(rounds)
    if max_noise_rate == None:
        max_noise_rate = [1] *len(rounds)

    p_th = data["p_th"] 
    nu = data["nu_fit"] 

    for i_r in range(len(rounds)):
        if select_rounds != None and not rounds[i_r] in select_rounds:
            continue # skip not selected rounds if 

        # distance mask
        d_mask = distances >= min_distance[i_r]
        # probability mask
        p_mask = (ps >= min_noise_rate[i_r]) & (ps <= max_noise_rate[i_r])

        plot_fssa_results(
            xs=[ps[p_mask]]*len(distances[d_mask]),
            ys=log_error_rates[d_mask, i_r][:,p_mask],
            yerrs=err_log_error_rates[d_mask, i_r][:,p_mask],
            distances=distances[d_mask],
            pc = p_th[i_r],
            nu = nu[i_r]
        )
    pass
