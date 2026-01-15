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
    # targeted qubits for generators of X-/Z-stabilizers
    targets_X, targets_Z = [],[] 
    # Plaquette = X-stabilizer
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
    # Site = Z-stabilizer
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

    # Initalize all physical qubits into a log. state by measurement of stabilizer gen. 
    targets_X, targets_Z = index_stab_targets(distance, offset)
    # Plaquette-/X-stabilizers
    circuit.append("H", index_X_ancilla, tag=marker_tag)
    for i_ancilla ,targets in zip(index_X_ancilla,targets_X):
        for i_target in targets:
            circuit.append("CNOT", [i_ancilla, i_target], tag=marker_tag)
    circuit.append("H", index_X_ancilla, tag=marker_tag)
    circuit.append("MR", index_X_ancilla, tag=marker_tag) # <- result (re)defines the codespace/generators of stabilizers! 
    # Site-/Z-stabilizers
    for i_ancilla ,targets in zip(index_Z_ancilla,targets_Z):
        for i_target in targets:
            circuit.append("CNOT", [i_target, i_ancilla], tag=marker_tag)
    circuit.append("MR", index_Z_ancilla, tag=marker_tag) # <- result (re)defines the codespace/generators of stabilizers!

    # Add Idendity operators to annotate the circuit
    circuit.append("I", index_physical, tag=final_tag)

    return circuit

def generate_steane_circuit(distance: int = 3):
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
    circuit = circuit_log_data + circuit_aux_0 + circuit_aux_p

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
    circuit.append("M",index_physical) 


    # Construct all relevant detectors 
    # Detectors should make the process of QEC simpler... right?!
    """
    Measurement catalogue:
    0           <   i   <   d*(d-1): |Psi> X-ancilla meas. (not needed)
    1*d*(d-1)   <=  i   < 2*d*(d-1): |Psi> Z-ancilla meas. => final result 
    2*d*(d-1)   <=  i   < 3*d*(d-1): |0> X-ancilla meas. => Z-error detector
    3*d*(d-1)   <=  i   < 4*d*(d-1): |0> Z-ancilla meas. (not needed)
    4*d*(d-1)   <=  i   < 5*d*(d-1): |+> X-ancilla meas. (not needed)
    5*d*(d-1)   <=  i   < 6*d*(d-1): |+> Z-ancilla meas. => X-error detector
    6*d*(d-1)                      <=  i   < 6*d*(d-1) +   (d**2 + (d-1)**2): |0> data qubit meas. => Z-error detector
    6*d*(d-1)+   (d**2 + (d-1)**2) <=  i   < 6*d*(d-1) + 2*(d**2 + (d-1)**2): |+> data qubit meas. => X-error detector 
    6*d*(d-1)+ 2*(d**2 + (d-1)**2) <=  i   < 6*d*(d-1) + 3*(d**2 + (d-1)**2): |Psi> data qubit meas. => final results
    """
    # Stim only support relative measurement references  
    def rel_meas(meas):
        # get relative inxex by absolute measurement index
        return meas - circuit.num_measurements

    # the code space is defined by the stabilizer generators, which we measured using the ancilla qubits
    # the previous measurement defines the sign of the stabilizer generator! 
    # We need to take those into account

    def add_detectors(circuit,targets,offset_def_ancilla,offset_measurements):
        for i, targets in enumerate(targets):
            sign_defining_ancilla = i + offset_def_ancilla 
            recombined_stabilizer = [t + offset_measurements for t in targets] 
            rel_meas_indices = [rel_meas(x) for x in [sign_defining_ancilla,*recombined_stabilizer]]
            circuit.append("DETECTOR",[stim.target_rec(x) for x in rel_meas_indices])
        return circuit

    offset_ancilla_psi_Z = 1*d*(d-1)  
    offset_ancilla_0_X = 2*d*(d-1) 
    offset_ancilla_p_Z = 5*d*(d-1)

    offset_data_psi = 6*d*(d-1) + 2*(d**2 + (d-1)**2)  
    offset_data_0 = 6*d*(d-1)
    offset_data_p = 6*d*(d-1) + (d**2 + (d-1)**2)

    targets_X, targets_Z = index_stab_targets(distance)
    # Detectors on |0> to detect Z-errors
    circuit = add_detectors(
        circuit,
        targets_X, 
        offset_ancilla_0_X,
        offset_data_0,
        )

    # Detectors on |p> to detect X-errors
    circuit = add_detectors(
        circuit,
        targets_Z, 
        offset_ancilla_p_Z,
        offset_data_p,
        )
    
    # Detector on |Psi> from observable measurement
    # TODO: I just need this if I want to make something fault tolerant, correct?
    # I might need to take the Z_stab measurement on |+>_L into account
    circuit = add_detectors(
        circuit,
        targets_Z,
        offset_ancilla_psi_Z,
        offset_data_psi,
        )

    # Logical Observable formed by combining all log. states (possible in odd distance log. qubits) 
    circuit.append("OBSERVABLE_INCLUDE",[stim.target_rec(-(i+1)) for i in range(distance**2 + (distance-1)**2)],0)


    return circuit
