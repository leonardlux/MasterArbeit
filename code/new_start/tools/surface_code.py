import numpy as np # type: ignore 
import stim # type: ignore


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
    # Vertices/Sites = X-stabilizer
    for row in range(d-1):
        for col in range(d):
            targets = [
                row*(2*d-1) + col, # top
                row*(2*d-1) + d + col, # right  
                (row+1)*(2*d-1) + col, # bottom
                row*(2*d-1) + d + col-1, # left 
                ]
            # BC: cutting out not existing qubits
            if col==0: # left edge 
                del targets[3] # del left qubit
            elif col==d-1: # right edge 
                del targets[1] # del right qubit 
            targets = np.array(targets) + offset 
            targets_X.append(targets)
    # Plaquette = Z-stabilizer
    for row in range(d):
        for col in range(d-1):
            targets = [
                (row-1)*(2*d-1) + d + col, # top 
                row*(2*d-1) + col+1, # right 
                row*(2*d-1) + d + col, # bottom 
                row*(2*d-1) + col, # left 
                ]
            # BC: cutting out not existing qubits
            if row==0: # top row 
                del targets[0]  
            elif row==d-1: # bottom row
                del targets[2] 
            targets = np.array(targets) + offset 
            targets_Z.append(targets)
    return targets_X, targets_Z

def index_log_Z(d,):
    qubit_z_measure = np.arange(d) * (2*d - 1) 
    return qubit_z_measure 

def index_log_X(d):
    qubit_x_measure = np.arange(d)
    return qubit_x_measure

# Stim only support relative measurement references  
def rel_meas(circuit, meas):
    # get relative inxex by absolute measurement index
    return meas - circuit.num_measurements

def add_detectors(circuit, targets, offsets_def_ancillas:list, offsets_measurements:list):
    """
    circuit: needed for num_measurements 
    targets: defines which qubits recombine to a stabilizer (by index)
    offsets_def_ancillas: offset of index of classical stabilizer measurements which define teh codespace
    offsets_measurements: offset of index of individual bits which should be recombined to stabilizer measurements
    """
    for i, targets in enumerate(targets):
        sign_defining_ancilla = [i + offset for offset in offsets_def_ancillas]  
        recombined_stabilizer = [t + offset for t in targets for offset in offsets_measurements] 
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

def generate_surface_code_log_qubit_circuit(distance: int = 3, offset=0, state: str = "0", final_tag: str = "" ):
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
    marker_tag = "l_qubit_init"

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

def generate_surface_code_circuit(distance: int = 3, state = "0", Z_stab: bool = True, X_stab: bool = True, ):
    """
    This functions generates a surface code ciruit with stabilizer readout.
    """
    if not state in ["0", "p"]:
        raise ValueError(f"does not know state {state}, expected: '0' for |0> or 'p' for |+>")
    d = distance
    # initialize
    surface_code_circ = generate_surface_code_log_qubit_circuit(
        distance,
        state=state,
        final_tag="psi_data",
        )
    # measure stabilizer 
    surface_code_circ = measure_surface_code_stabilizer(
        distance,
        surface_code_circ,
    ) 
    # Measure observable by measuremnt of logical data from main qubit |Psi>
    index_physical, _, _ = index_qubits_surface_code(distance)
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
    
    targets = []
    i_ls = index_log_Z(d) #TODO: adapt for arbitray intput state!
    for i_qubit in i_ls:
        targets += [d**2 + (d-1)**2 - i_qubit] 

    surface_code_circ.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-(i)) for i in targets ],
        0, # 0. log measurement
        )

    return surface_code_circ

#TODO: adapt to arbtriary inital state!
def generate_steane_circuit(distance: int = 3, ft_stab_detector: bool = True, rounds: int = 1):
    """
    This function generates the the steane type EC.
    
    :param distance: Distance of the used log. qubits 
    :type distance: int
    """
    d = distance
    offset_per_log_qubit = (d**2 + (d-1)**2 + 2*d*(d-1)) # num. of all phy. qubits per log. qubit
    # init. all qubits in correct state!
    circuit_log_data = generate_surface_code_log_qubit_circuit(
        distance, 
        offset=0,
        state="0",
        final_tag="psi_data"
        )
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
    # TODO: I just need this if I want to make something fault tolerant, correct?
    # TODO: Actually destroys pymatching if this detector is active!! WHY?! -> because pymatching requires one version for each error (overcomplete?) (deconstruct error thing)
    # I might need to take the Z_stab measurement on |+>_L into account
    if ft_stab_detector:
        circuit = add_detectors(
            circuit,
            targets_Z,
            [os_ancilla_psi_Z],
            [os_data_psi],
            )

    # Logical Observable (check in notes) 
    targets = []
    i_ls = index_log_Z(d) #TODO: adapt for arbitray intput state!
    for i_qubit in i_ls:
        targets += [d**2 + (d-1)**2 - i_qubit] 
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(-(i)) for i in targets ],
        0, # 0. log measurement
        )

    return circuit