import numpy as np # type: ignore 
import stim # type: ignore
import numba


def index_qubits_surface_code(distance: int = 3, offset: int = 0):
    d = distance
    if d%2!=1:
        raise ValueError(f"Odd number expected, got {d}")
    n_current = 0
    # number of physical qubits: qubits on (horizontal) + (vertical) edges
    n_physical = d**2 + (d-1)**2
    index_physical = np.arange(n_current,n_current+n_physical) + offset
    n_current += n_physical
    # plaquette stabilizer (X)
    n_X_stab = d * (d-1) 
    index_X_ancilla = np.arange(n_current,n_current+n_X_stab) + offset
    n_current += n_X_stab 
    # site stabilizer (Z)
    n_Z_stab = d * (d-1) 
    index_Z_ancilla = np.arange(n_current,n_current+n_Z_stab) + offset
    n_current += n_Z_stab 

    return index_physical, index_X_ancilla, index_Z_ancilla 

def index_stab_targets(distance: int = 3, offset: int = 0, tag: str = ""):
    if distance%2!=1:
        raise ValueError(f"Odd number expected, got {distance}")
    d = distance
    # targeted qubits for X-/Z-stabilizers
    targets_X, targets_Z = [],[] 
    # N=0, E=1, S=2, W=3 
    reorder_X = [2,3,1,0]
    reorder_Z = [2,1,3,0]
    # Vertices/Sites = X-stabilizer
    for row in range(d-1):
        for col in range(d):
            targets = np.array([
                row*(2*d-1) + col, # top
                row*(2*d-1) + d + col, # right  
                (row+1)*(2*d-1) + col, # bottom
                row*(2*d-1) + d + col-1, # left 
                ])
            targets = list(targets[reorder_X])
            # BC: cutting out not existing qubits
            if col==0: # left edge 
                del targets[reorder_X.index(3)] # del left qubit
            elif col==d-1: # right edge 
                del targets[reorder_X.index(1)] # del right qubit 
            targets = np.array(targets) + offset 
            targets_X.append(targets)
    # Plaquette = Z-stabilizer
    for row in range(d):
        for col in range(d-1):
            targets = np.array([
                (row-1)*(2*d-1) + d + col, # top 
                row*(2*d-1) + col+1, # right 
                row*(2*d-1) + d + col, # bottom 
                row*(2*d-1) + col, # left 
                ])
            targets = list(targets[reorder_Z])
            # BC: cutting out not existing qubits
            if row==0: # top row 
                del targets[reorder_Z.index(0)]  
            elif row==d-1: # bottom row
                del targets[reorder_Z.index(2)] 
            targets = np.array(targets) + offset 
            targets_Z.append(targets)
    return targets_X, targets_Z

# Multi_row_logical needs to be identical for both X and Z!
@numba.njit
def index_log_Z(d, multi_row_logical:bool = False):
    cols = d if multi_row_logical else 1

    qubit_z_measure = np.empty(cols * d, dtype=np.int64)
    k = 0
    for i in range(cols):
        for j in range(d):
            qubit_z_measure[k] = i + j * (2*d - 1)
            k += 1
    return qubit_z_measure 

@numba.njit
def index_log_X(d, multi_row_logical:bool = False):
    rows = d if multi_row_logical else 1

    qubit_x_measure = np.empty(rows* d, dtype=np.int64)
    k = 0
    for i in range(rows):
        for j in range(d):
            qubit_x_measure[k] = j + i * (2*d - 1)
            k += 1
    return qubit_x_measure

# Stim only support relative measurement references  
def rel_meas(circuit, meas):
    # get relative inxex by absolute measurement index
    return meas - circuit.num_measurements

def add_detectors(circuit, targets, offsets_stabilizer_measurements:list, offsets_measurements_to_be_stabilizer:list):
    """
    circuit: needed for num_measurements 
    targets: defines which qubits recombine to a stabilizer (by index)
    offsets_stabilizer_measurements: offset of index of classical stabilizer measurements which should be included in detector 
    offsets_measurements: offset of index of individual bits which should be recombined to stabilizer measurements
    """
    for i, targets in enumerate(targets):
        sign_defining_ancilla = [i + offset for offset in offsets_stabilizer_measurements]  
        recombined_stabilizer = [t + offset for t in targets for offset in offsets_measurements_to_be_stabilizer] 
        rel_meas_indices = [rel_meas(circuit,x) for x in [*sign_defining_ancilla,*recombined_stabilizer]]
        circuit.append("DETECTOR",[stim.target_rec(x) for x in rel_meas_indices])
    return circuit

def measure_surface_code_stabilizer(distance, circuit, offset=0, tag=""):

    _, index_X_ancilla, index_Z_ancilla = index_qubits_surface_code(distance, offset)
    # Initalize all physical qubits into a log. state by measurement of stabilizer
    targets_X, targets_Z = index_stab_targets(distance, offset)
    # Site-/X-stabilizers
    circuit.append("H", index_X_ancilla, tag=tag)
    for i_ancilla ,targets in zip(index_X_ancilla,targets_X):
        for i_target in targets:
            circuit.append("CNOT", [i_ancilla, i_target], tag=tag)
    circuit.append("H", index_X_ancilla, tag=tag)
    circuit.append("MR", index_X_ancilla, tag=tag) # <- result (re)defines the codespace/generators of stabilizers! 
    # Plaquette-/Z-stabilizers
    for i_ancilla ,targets in zip(index_Z_ancilla,targets_Z):
        for i_target in targets:
            circuit.append("CNOT", [i_target, i_ancilla], tag=tag)
    circuit.append("MR", index_Z_ancilla, tag=tag) # <- result (re)defines the codespace/generators of stabilizers! 

    return circuit

def generate_surface_code_log_qubit_circuit(distance: int = 3, offset=0, state: str = "0", final_tag: str = "", marker_tag="l_qubit_init" ):
    """
    This function adds a log qubit, encoded in surface code, to the circuit. 
    In this process it:
    1. resets the needed physical qubits
    2. uses ancillas to measure the qubits in the correct bases
    
    :param distance: distance of the surface code 
    :type distance: int
    :param offset: starting index of the first phy. qubit 
    :param state: string which is either "0" for |0> or "p" for |+> 
    :type state: str
    :param final_tag: tag that will be assigned to physical data index after init 
    :type final_tag: str
    """
    circuit = stim.Circuit()

    index_physical, index_X_ancilla, index_Z_ancilla = index_qubits_surface_code(distance, offset)
    # init qubits by resetting them to 0 (technicly useless, but goal here is to do the implicit obvious)
    if state == "0": # |0>
        circuit.append("R", index_physical, tag=marker_tag)
    elif state == "p": # |+>
        circuit.append("R", index_physical, tag=marker_tag)
        circuit.append("H", index_physical, tag=marker_tag)
    else:
        raise ValueError(f"does not know state {state}, expected: '0' for |0> or 'p' for |+>")
    circuit.append("R", index_X_ancilla, tag=marker_tag)
    circuit.append("R", index_Z_ancilla, tag=marker_tag)

    # initalize by measuring stabilizer (the resulting measurements define the codespace)
    circuit = measure_surface_code_stabilizer(
        distance,
        circuit,
        offset,
        tag = marker_tag,
    )

    # Add Idendity operators to annotate the end of log qubit init in the circuit
    circuit.append("I", index_physical, tag=final_tag)

    return circuit

def generate_simple_surface_code_circuit(distance: int = 3, observable: int = "Z", Z_stab: bool = True, X_stab: bool = True, ):
    """
    This functions generates a surface code ciruit with stabilizer readout.
    Note that Z_stab, and X_stab only switch the detectors on or off, the ciruit layout is unchanged.
    """
    d = distance
    if observable == "Z":
        init_log_state = "0" 
        i_logical_measurement_targets = index_log_Z(d)
    elif observable == "X":
        init_log_state = "p"
        i_logical_measurement_targets = index_log_X(d)
    else:
        raise ValueError(f"unkown observable value: {observable}")
    # initialize
    surface_code_circ = generate_surface_code_log_qubit_circuit(
        distance,
        state=init_log_state,
        final_tag="psi_data",
        )
    # measure stabilizer 
    surface_code_circ = measure_surface_code_stabilizer(
        distance,
        surface_code_circ,
    ) 
    # Measure observable by measuremnt of logical data from main qubit |Psi>
    index_physical, _, _ = index_qubits_surface_code(distance)
    if observable == "X":
        surface_code_circ.append("H",index_physical, tag="obs_change_of_basis")
    surface_code_circ.append("M",index_physical,tag="obs_flip_measure") 

    """
    Measurement catalogue:
    dc = define codespace
    0           <   i   <   d*(d-1): |Psi> X-stab. meas.    -(dc)-> steane X-stab = Z-error detector  
    1*d*(d-1)   <=  i   < 2*d*(d-1): |Psi> Z-stab. meas.    -(dc)-> steane Z-stab = X-error detector 
    2*d*(d-1)   <=  i   < 3*d*(d-1): |Psi> X-stab. meas.    -> steane X-stab = Z-error detector 
    3*d*(d-1)   <=  i   < 4*d*(d-1): |Psi> Z-stab. meas.    -> steane Z-stab = X-error detector
    4*d*(d-1)   <=  i   < 4*d*(d-1) + (d^2 + (d-1)^2): |Psi> data qubit meas. -> Z-observable => final obs 
    """
    # generate detectors    
    offset_ancilla_psi_X_dc = 0
    offset_ancilla_psi_Z_dc = 1*d*(d-1)  
    offset_ancilla_psi_X    = 2*d*(d-1)  
    offset_ancilla_psi_Z    = 3*d*(d-1)  

    targets_X, targets_Z = index_stab_targets(distance)
    # X-stabilizer 
    if X_stab:
        surface_code_circ = add_detectors(
            surface_code_circ,
            targets_X, 
            [offset_ancilla_psi_X_dc, offset_ancilla_psi_X],
            [], # we do not recombine measurements into stabilizers
            )

    # Z-stabilizer 
    if Z_stab:
        surface_code_circ= add_detectors(
            surface_code_circ,
            targets_Z, 
            [offset_ancilla_psi_Z_dc, offset_ancilla_psi_Z],
            [], # we do not recombine measurements into stabilizers
            )    

    # Measure observable 
    targets = []
    for i_qubit in i_logical_measurement_targets:
        targets += [d**2 + (d-1)**2 - i_qubit] 

    surface_code_circ.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-(i)) for i in targets ],
        0, # 0. log measurement
        )

    return surface_code_circ

def generate_ft_surface_code_circuit(distance: int = 3, rounds: int = 1, observable: int = "Z", ft_stab: bool = True):
    """
    This functions generates a surface code ciruit with d-times stabilizer readout for each round.
    Note that Z_stab, and X_stab only switch the detectors on or off, the ciruit layout is unchanged.
    """
    new_round_tag = "new_round"
    new_round_tag = "new_stab"
    d = distance
    if observable == "Z":
        init_log_state = "0" 
        i_logical_measurement_targets = index_log_Z(d)
    elif observable == "X":
        init_log_state = "p"
        i_logical_measurement_targets = index_log_X(d)
    else:
        raise ValueError(f"unkown observable value: {observable}")
    # initialize
    surface_code_circ = generate_surface_code_log_qubit_circuit(
        distance,
        state=init_log_state,
        final_tag="psi_data",
        )
    index_physical, _, _ = index_qubits_surface_code(distance)
    for i_r in range(rounds):
        if i_r != 0:
            pass 
            # surface_code_circ.append("I", index_physical, tag=new_round_tag) # not sure if I should include this?
        # each round
        for i_d in range(distance):
            # in each rounds we need to measure the stabilizer d-times
            # measure stabilizer 
            surface_code_circ = measure_surface_code_stabilizer(
                distance,
                surface_code_circ,
                tag="faulty"
            ) 
    # Measure observable by measuremnt of logical data from main qubit |Psi>
    if observable == "X":
        surface_code_circ.append("H",index_physical, tag="obs_change_of_basis")
    surface_code_circ.append("M",index_physical,tag="obs_flip_measure") 

    n_stab = d*(d-1)
    n_round = 2*d*n_stab  
    n_qubits = d**2 + (d-1)**2 
    """
    Measurement catalogue:
    pf = Pauli frame (relevant) 
    # Initial:
    0           <   i   <   n_stab: |Psi> X-stab. meas.    -(pf)-> steane X-stab = Z-error detector  
    1*n_stab    <=  i   < 2*n_stab: |Psi> Z-stab. meas.    -(pf)-> steane Z-stab = X-error detector 
    # i_r : QEC Rounds max: rounds
    # i_d : Stabilizer Repetition (d times)
    2*n_stab + i_d*2*n_stab + i_r*2*d*n_stab  <=  i   < 3*n_stab + i_d*2*n_stab + i_r*2*d*n_stab: |Psi> X-stab. meas.    -> steane X-stab = Z-error detector 
    3*n_stab + i_d*2*n_stab + i_r*2*d*n_stab  <=  i   < 4*n_stab + i_d*2*n_stab + i_r*2*d*n_stab: |Psi> Z-stab. meas.    -> steane Z-stab = X-error detector

    # Observable
    (rounds * d + 1) * 2*n_stab  <=  i   < (rounds * d + 1) * 2*n_stab + n_qubits: |Psi> data qubit meas. -> Z-observable => final obs 
    """
    # generate detectors    
    offset_ancilla_psi_X_pf = 0
    offset_ancilla_psi_Z_pf = 1*d*(d-1)  
    offset_ancilla_psi_X    = 2*d*(d-1)  
    offset_ancilla_psi_Z    = 3*d*(d-1)  
    
    offset_obs_psi = (rounds * d + 1) * 2*n_stab  

    targets_X, targets_Z = index_stab_targets(distance)

    prev_offset_ancilla_X = offset_ancilla_psi_X_pf
    prev_offset_ancilla_Z = offset_ancilla_psi_Z_pf
    for i_r in range(rounds):
        for i_d in range(distance):
            current_offset_due_to_loop = i_r * n_round + i_d * 2 * n_stab  
            # X-stabilizer
            surface_code_circ = add_detectors(
                circuit=surface_code_circ,
                targets=targets_X, 
                offsets_stabilizer_measurements=[
                    offset_ancilla_psi_X + current_offset_due_to_loop,
                    prev_offset_ancilla_X,
                ],
                offsets_measurements_to_be_stabilizer=[], # we do not recombine measurements into stabilizers, we measure stabilizer directly!
                )
            # Z-stabilizer 
            surface_code_circ= add_detectors(
                circuit=surface_code_circ,
                targets=targets_Z, 
                offsets_stabilizer_measurements=[
                    offset_ancilla_psi_Z + current_offset_due_to_loop,
                    prev_offset_ancilla_Z,
                ],
                offsets_measurements_to_be_stabilizer=[], # we do not recombine measurements into stabilizers, we measure stabilizer directly!
                )    
            prev_offset_ancilla_X = offset_ancilla_psi_X + current_offset_due_to_loop
            prev_offset_ancilla_Z = offset_ancilla_psi_Z + current_offset_due_to_loop

    # Detector on |Psi> from observable measurement
    if ft_stab: 
        if observable == "Z":
            ft_targets = targets_Z
            prev_measurement = prev_offset_ancilla_Z # already xor with prev. measurement
        else:
            ft_targets = targets_X
            prev_measurement = prev_offset_ancilla_X
        surface_code_circ = add_detectors(
            surface_code_circ,
            ft_targets,
            [prev_measurement],
            [offset_obs_psi],
            )
    # Measure observable 
    targets = []
    for i_qubit in i_logical_measurement_targets:
        targets += [d**2 + (d-1)**2 - i_qubit] 

    surface_code_circ.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-(i)) for i in targets ],
        0, # 0. log measurement
        )

    return surface_code_circ

def generate_steane_circuit(distance: int = 3, rounds: int = 1, observable: str = "Z",ft_stab_detector: bool = True):
    """
    This function generates the the steane type EC.
    
    :param distance: Distance of the used log. qubits 
    :type distance: int
    :param observable: choice of observable measurement on data qubit  

    oberservable influences inital state: 
        "Z"->"|0>"
        "X"->"|+>"
    """
    d = distance
    if observable == "Z":
        init_log_state = "0" 
        i_logical_measurement_targets = index_log_Z(d)
    elif observable == "X":
        init_log_state = "p"
        i_logical_measurement_targets = index_log_X(d)
    else:
        raise ValueError(f"unkown observable value: {observable}")

    offset_per_log_qubit = (d**2 + (d-1)**2 + 2*d*(d-1)) # num. of all phy. qubits per log. qubit
    # init. 
    circuit_log_data = generate_surface_code_log_qubit_circuit(
        distance, 
        offset=0,
        state=init_log_state,
        final_tag="psi_data"
        )
    # init ancilla qubits
    circuit_aux_0 = generate_surface_code_log_qubit_circuit(
        distance, 
        offset = offset_per_log_qubit,
        state="0",
        final_tag="0_data"
        )
    circuit_aux_p = generate_surface_code_log_qubit_circuit(
        distance, 
        offset = 2*offset_per_log_qubit,
        state="p",
        final_tag="p_data"
        )
    circuit = circuit_log_data
    for round in range(rounds):
        # add components over and over
        circuit = circuit + circuit_aux_0 + circuit_aux_p  
        # Steane Connections/Entanglement! 
        index_physical, _, _ = index_qubits_surface_code(distance)
        # entangle C|0>NOT|Psi> 
        for i in index_physical:
            # CxNOTy: "CNOT",[x,y]
            circuit.append("CNOT", (offset_per_log_qubit + i, i))
        # measure:
        circuit.append("H", index_physical + offset_per_log_qubit) 
        # note the Hadamard is NOT! a log. Hadamard (not transversal)
        circuit.append("MR",index_physical + offset_per_log_qubit) 

        # entangle C|Psi>NOT|p> 
        for i in index_physical:
            # CxNOTy: "CNOT",[x,y]
            circuit.append("CNOT", (i, 2*offset_per_log_qubit + i))
        # measure:
        circuit.append("MR",index_physical + 2*offset_per_log_qubit) 

    # Measure observable by measuremnt of logical data from main qubit |Psi>
    if observable == "X":
        circuit.append("H",index_physical, tag="obs_change_of_basis")
    circuit.append("M",index_physical,tag="obs_flip_measure") 

    # Construct all relevant detectors 
    # Detectors should make the process of QEC simpler... right?!
    """
    ("dc" = "defines codespace")
    r: number of rounds 
    Number of measurements
    n_stab  = d*(d-1)
    n_qubit = d**2 + (d-1)**2
    n_round  = 4 * n_stab + 2 * n_qubit 

    Measurement catalogue:
        0          <   i   <   n_stab: |Psi> X-stab. meas.    -(dc)-> steane X-stab = Z-error detector 
        1*n_stab   <=  i   < 2*n_stab: |Psi> Z-stab. meas.    -(dc)-> steane Z-stab = X-error detector & FT check 
    - start of new round (each round offset by: offset = n_round * r + 2*n_stab)
        0          <=  i   < 1*n_stab: |0>   X-stab. meas.    -(dc)-> steane X-stab = Z-error detector
        1*n_stab   <=  i   < 2*n_stab: |0>   Z-stab. meas. (not needed)
        2*n_stab   <=  i   < 3*n_stab: |+>   X-stab. meas. (not needed)
        3*n_stab   <=  i   < 4*n_stab: |+>   Z-stab. meas.    -(dc)-> steane Z-stab = X-error detector
        4*n_stab           <=  i < 4*n_stab +   n_qubit: |0> data qubit meas. -> steane X-stab => Z-error detector
        4*n_stab + n_qubit <=  i < 4*n_stab + 2*n_qubit: |+> data qubit meas. -> steane Z-stab => X-error detector 
    - end of round 
        2*n_stab + r*n_loop <=  i   < 2*n_stab + reps*n_loop + n_qubit: |Psi> data qubit meas. -> Z-observable => final obs & FT check
    """
    # the code space is defined by the stabilizer generators, which we measured using the ancilla qubits
    # the previous measurement defines the sign of the stabilizer generator! 
    # We need to take those into account

    n_stab  = d*(d-1)
    n_qubit = d**2 + (d-1)**2
    n_round = 4 * n_stab + 2 * n_qubit

    # offsets = os
    # X = X-stab; Z = Z-stab
    os_ancilla_psi_X = 0
    os_ancilla_psi_Z = 1 * n_stab
    os_loop = 2 * n_stab

    # loop rel offsets
    os_ancilla_0_X   = 0 * n_stab 
    os_ancilla_p_Z   = 3 * n_stab
    os_data_0 = 4 * n_stab
    os_data_p = 4 * n_stab + n_qubit 

    # end offset
    os_data_psi = 2 * n_stab + rounds * n_round 

    targets_X, targets_Z = index_stab_targets(distance)
    for round in range(rounds):
        # Detectors on |0> to detect Z-errors
        cur_os = os_loop + n_round * round
        circuit = add_detectors(
            circuit,
            targets_X, 
            [
                cur_os + os_ancilla_0_X, 
                os_ancilla_psi_X,
            ],
            [
                cur_os + os_data_0,
            ],
            )

        # Detectors on |p> to detect X-errors
        circuit = add_detectors(
            circuit,
            targets_Z, 
            [
                cur_os + os_ancilla_p_Z, 
                os_ancilla_psi_Z,
            ],
            [
                cur_os + os_data_p,
            ],
            )
    
    # Detector on |Psi> from observable measurement
    if ft_stab_detector:
        if observable == "Z":
            ft_targets = targets_Z
            ft_ancilla = os_ancilla_psi_Z
        else:
            ft_targets = targets_X
            ft_ancilla = os_ancilla_psi_X
        circuit = add_detectors(
            circuit,
            ft_targets,
            [ft_ancilla],
            [os_data_psi],
            )

    # Logical Observable (check in notes) 
    targets = []
    for i_qubit in i_logical_measurement_targets:
        targets += [d**2 + (d-1)**2 - i_qubit] 
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-(i)) for i in targets ],
        0, # 0. log measurement
        )

    return circuit

def config_to_circ_func(config):
    value = config["circuit"]["type"]
    if  value == "steane":
        return generate_steane_circuit
    elif value == "surface":
        return generate_ft_surface_code_circuit
    else:
        raise ValueError(f"unkown config parameter: circ, type: {value}")