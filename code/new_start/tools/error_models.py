import stim  # type: ignore

def add_noise(
        circuit,
        noise_model,
        ):
    """
    add_noise adds noised to an existing circuit, based on the parameters of the noise_model and the structure of the circuit.

    :param circuit: a stim.Circuit() obj. 
    :param noise_model: a dict with error parameters 
    """
    noisy_circuit = stim.Circuit()
    # Hardcoded to ignore the initialization process of qubits
    faultless_tags = {
        "l_qubit_init",
    }
    # I use the "I" Operation to add some inital error onto my model
    for  circ_instr in circuit:
        # circ_instr = circuit instruction
        if circ_instr.tag in faultless_tags: 
            # Skip all possible errors 
            noisy_circuit.append(circ_instr)
            continue
        # Errors before operations
        for key in noise_model:
            if circ_instr.name in noise_model[key]["operator"] and noise_model[key]["error_position"]=="before": 
                if not "specific_tag" in noise_model[key] or circ_instr.tag in noise_model[key]["specific_tag"]:
                    noisy_circuit.append(
                        noise_model[key]["error"],
                        circ_instr.targets_copy(), 
                        noise_model[key]["noise"],
                    )
        # Orignial operation
        noisy_circuit.append(circ_instr)
        # Errors after operation
        for key in noise_model:
            if circ_instr.name in noise_model[key]["operator"] and noise_model[key]["error_position"]=="following": 
                if not "specific_tag" in noise_model[key] or circ_instr.tag in noise_model[key]["specific_tag"]:
                    noisy_circuit.append(
                        noise_model[key]["error"],
                        circ_instr.targets_copy(), 
                        noise_model[key]["noise"],
                    )
    return noisy_circuit

# Automatic construct  
def construct_basic_noise_model(noise: float, X_errors = True, Z_errors = False):
    """
    construct_basic_error_model, constructs a noise model where there are only error after initilazation of the main data qubit.
    
    :param noise: define the error rate of the occuring errors 
    :type noise: float
    """
    noise_model = {} 
    if X_errors:
        noise_model["inital_X_errors"] = { 
            "operator": {"I"}, 
            "error_position": "following", 
            "error": "X_ERROR", 
            "noise": noise, 
            "specific_tag": {"psi_data"}, 
        }
    if Z_errors:
        noise_model["inital_Z_errors"] = {
            "operator": {"I"}, 
            "error_position": "following", 
            "error": "Z_ERROR", 
            "noise": noise, 
            "specific_tag": {"psi_data"}, 
        }
    return noise_model

"""
Additionaly:
Some examples
"""

# Most basic example of a noise model
noise_model = {
    "<some name>": {
        "operator": {"H"}, # set {} of str descriping the stim instructions/targets beeing targeted
        "error_position": "before", # or "following" in regard to the orignial instruction
        "error": "X_ERROR", # str with name of the stim error instruction
        "noise": 0.1, # noise rate of the error  
        "specific_tag": {"psi_data"}, # specific tag of targeted instruction
    }
}


# Example of a noise model
# Parameters stolen from the paper "Demonstration of fault-tolerant Steane quantum error correction"
noise = 0.01
noise_model = {
    "single_qubit_gate": {
        "operator": {"H"},
        "error_position": "following",
        "error": "DEPOLARIZE1",
        "noise": noise,
    },
    "two_qubit_gate": {
        "operator": {"CX"},
        "error_position": "following",
        "error": "DEPOLARIZE2",
        "noise": noise,
    },
    "measurement": {
        "operator": {"M","MR"},
        "error_position": "before",
        "error": "X_ERROR",
        "noise": 0.003,
    },
    "initialize": {
        "operator": {"I"}, # Trick I use because init algo is out of scope! 
        "error_position": "following",
        "error": "X_ERROR",
        "noise": 0.003,
    },
}
