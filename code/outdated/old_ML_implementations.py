# kill this!
def count_logical_errors_using_ML(
        circuit, 
        num_shots: int, 
        distance: int, 
        error_rate: float, 
        rounds: int,
        probability: bool = False,
        **kwargs, # just here to ignore the stuff other logical error counter need
    ) -> float:
    d = distance
    p = error_rate 
    
    detection_events, observable_flips = sample_ciruit(circuit, num_shots) 

    # init ML decoder (only X errors so far)
    num_errors = 0
    # iterating over each shot!
    for obs, d_event in zip(observable_flips,detection_events): 
        x_stab_syndrome, z_stab_syndrome, =  split_syndrome(d, d_event)
        pred_log, _ = decode_half_syndrome_aron(
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

# kill this!
def dead_ML_FT(
        circuit, 
        num_shots: int, 
        distance: int, 
        error_rate: float, 
        rounds: int,
        probability: bool = False,
        **kwargs, # just here to ignore the stuff other logical error counter need
    ) -> float:
    # Also works for basic noise model! xor syndrome == 0
    d = distance
    p_eff, _ = uncorr_eff_noise(error_rate)
    p_eff = error_rate

    
    detection_events, observable_flips = sample_ciruit(circuit, num_shots) 

    x_synds, z_synds, ft_synds = split_syndromes(d,detection_events)

    # init ML decoder (only X errors so far)
    num_errors = 0
    # iterating over each shot! one can use paralism here! (if I design it a bit nicer!)
    for obs, z_synd, ft_synd in zip(observable_flips, z_synds, ft_synds): 
        # normal syndrome decoding

        pred, c_f = decode_half_syndrome_aron(
            d,
            p_eff,
            z_synd,
        )
        # ft syndrome decoding
        xor_ft_synd = xor_ft_syndrome(ft_synd, z_synd)
        pred_ft, c_f_ft = decode_half_syndrome_aron(
            d,
            p_eff,
            xor_ft_synd,
        )

        pred_obs = pred ^ c_f ^ pred_ft ^ c_f_ft

        if obs[0] != pred_obs:
            num_errors += 1
    if probability:
        return num_errors/num_shots
    return num_errors 
