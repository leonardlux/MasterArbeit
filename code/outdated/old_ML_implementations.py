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

def decode_mwpm_steane(d, p, syndromes, z_stab:bool = True, noise_model:str =  "basic"):
    """
    d: distance 
    p: error rate 
    syndromes: list of generated syndromes
    z_stab: True -> calculate prediciton for log X (|0> init state)
            False-> calculate prediction for log Z (|+> init state) <=> calcuates x_stab
    noise_model: defines which noise model the decoder assumes

    decoded using effective circuit and noise model!!
    treats X and Z noise independet 
    returns list of predictions for observable
    if both true returns list of list! 
    """
    x_stab_syndromes, z_stab_syndromes, ft_syndromes = split_syndromes(d, syndromes) 

    # z_stab==True: Z-Stabilizer <=> X-Errors
    # z_stab==False: X-Stabilizer <=> Z-Errors
    stab_syndromes = z_stab_syndromes if z_stab else x_stab_syndromes

    matcher = gen_mwpm_matcher(d, p, z_stab, noise_model)
    log_predictions = matcher.decode_batch(stab_syndromes)

    if noise_model == "circ":
        # assuming same error probability for ft error (TODO correct this!)
        ft_syndromes = np.logical_xor(ft_syndromes,stab_syndromes) # pauli frame tracking 
        ft_matcher = gen_mwpm_matcher(d,p) 
        ft_predicitons = ft_matcher.decode_batch(ft_syndromes)
        log_predictions = log_predictions^ft_predicitons

    return log_predictions

def decode_mwpm_reps(d, p, reps, syndromes, z_stab:bool = True,):
    """
    d: distance 
    p: error rate (assumes circuit lvl noise and uniform noise probability everywhere)
    syndromes: list of generated syndromes
    z_stab: True -> calculate prediciton log X (|0> state relevant)
            False-> calculate prediction log Z (|+> state relevant) <=> calcuates x_stab
    noise_model: defines which noise model the decoder assumes

    decoded using effective circuit and noise model!!
    treats X and Z noise independet 
    returns list of predictions for observable
    if both true returns list of list! 
    """
    # pauli frame tracked syndromes
    px_synd, pz_synd, pft_synd = split_and_xor_syndrome(d, reps, syndromes, z_stab)

    # z_stab==True: Z-Stabilizer <=> X-Errors
    if z_stab: 
        stab_syndromes = pz_synd 
    # z_stab==False: X-Stabilizer <=> Z-Errors
    elif not z_stab:
        stab_syndromes = px_synd 

    shots = len(stab_syndromes[0])
    log_predictions = np.zeros((reps+1,shots,1)) # reps +1 for ft-check

    # Steane syndromes
    matcher = gen_mwpm_matcher(d, p, z_stab, noise_model="circ_lvl")
    for rep in range(reps):
        log_predictions[rep] = matcher.decode_batch(stab_syndromes[rep])
    # FT syndrome
    ft_matcher = gen_mwpm_matcher(d, p, z_stab, noise_model="circ_lvl")
    log_predictions[reps] = ft_matcher.decode_batch(pft_synd)

    final_predictions = np.zeros((shots))
    for i in range(len(log_predictions)):
        final_predictions = np.logical_xor(final_predictions,log_predictions[i])

    return final_predictions
# TODO correct error probability for FT syndrome (can I calc that?) 
# TODO build iterative multi round decoder! 
# clean out MWPM
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
        detection_events, observable_flips = sample_ciruit(circuit, num_shots) 

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

