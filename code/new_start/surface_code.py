import numpy as np 
import stim


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

def index_stab_targets(distance: int = 3, offset: int = 0):
    if distance%2!=1:
        raise ValueError(f"Odd number expected, got {distance}")
    d = distance
    # targeted qubits for generators of X-/Z-stabilizers
    targets_X, targets_Z = [],[] 
    # Plaquette = X-stabilizer
    for row in range(d-1):
        for col in range(d):
            targets = [
                row*(2*d-1) + col + offset, # top
                row*(2*d-1) + d + col + offset, # right  
                (row+1)*(2*d-1) + col + offset , # bottom
                row*(2*d-1) + d + col-1 + offset, # left 
                ]
            # BC: cutting out not existing qubits
            if col==0: # left edge 
                del targets[3] # del left qubit
            elif col==d-1: # right edge 
                del targets[1] # del right qubit 
            targets_X.append(targets)
    # Site = Z-stabilizer
    for row in range(d):
        for col in range(d-1):
            targets = [
                (row-1)*(2*d-1) + d + col + offset, # top 
                row*(2*d-1) + col+1 + offset, # right 
                row*(2*d-1) + d + col + offset, # bottom 
                row*(2*d-1) + col + offset, # left 
                ]
            # BC: cutting out not existing qubits
            if row==0: # top row 
                del targets[0]  
            elif row==d-1: # bottom row
                del targets[2] 
            targets_Z.append(targets)
    return targets_X, targets_Z

def generate_surface_code_circuit(distance: int = 3, offset=0, state: str ="0"):
    circuit = stim.Circuit()

    index_physical, index_X_ancilla, index_Z_ancilla = index_qubits_surface_code(distance, offset)
    # init qubits by resetting them to 0
    if state == "0": # |0>
        circuit.append("R", index_physical, tag="init")
    elif state == "p": # |+>
        circuit.append("R", index_physical, tag="init")
        circuit.append("H", index_physical, tag="init")
    else:
        raise ValueError(f"does not know state {state}, expected: '0' for |0> or 'p' for |+>")
    circuit.append("R", index_X_ancilla, tag="init")
    circuit.append("R", index_Z_ancilla, tag="init")

    # Init into a logical state by measurement of stabilizers
    targets_X, targets_Z = index_stab_targets(distance, offset)
    # Plaquette-/X-stabilizers
    circuit.append("H", index_X_ancilla)
    for i_ancilla ,targets in zip(index_X_ancilla,targets_X):
        for i_target in targets:
            circuit.append("CNOT", [i_ancilla, i_target])
    circuit.append("H", index_X_ancilla)
    circuit.append("MR", index_X_ancilla) # <- redefines the codespace/generators of stabilizers! 
    # Site-/Z-stabilizers
    for i_ancilla ,targets in zip(index_Z_ancilla,targets_Z):
        for i_target in targets:
            circuit.append("CNOT", [i_target, i_ancilla])
    circuit.append("MR", index_Z_ancilla) # <- redefines the codespace/generators of stabilizers!
    return circuit

def generate_steane_set_up(distance: int = 3):
    d = distance
    offset_per_log_qubit = (d**2 + (d-1)**2 + 2*d*(d-1)) # num. of all log. qubits
    # init all qubits!
    circuit_log_data = generate_surface_code_circuit(
        distance, 
        offset=0,
        )
    circuit_aux_0 = generate_surface_code_circuit(
        distance, 
        offset = offset_per_log_qubit,
        )
    circuit_aux_p = generate_surface_code_circuit(
        distance, 
        offset = 2*offset_per_log_qubit,
        state="p",
        )
    total_circuit = circuit_log_data + circuit_aux_0 + circuit_aux_p

    # Steane Connections! 
    index_physical, _, _ = index_qubits_surface_code(distance)
    # entangle C|0>NOT|Psi> 
    for i in index_physical:
        # CxNOTy: "CNOT",[x,y]
        total_circuit.append("CNOT", (offset_per_log_qubit + i, i))
    # measure:
    total_circuit.append("H", index_physical + offset_per_log_qubit) 
    # note the Hadamard is NOT! a log. Hadamard (not transversal)
    total_circuit.append("MR",index_physical + offset_per_log_qubit) 

    # entangle C|Psi>NOT|p> 
    for i in index_physical:
        # CxNOTy: "CNOT",[x,y]
        total_circuit.append("CNOT", (i, 2*offset_per_log_qubit + i))
    # measure:
    total_circuit.append("MR",index_physical + 2*offset_per_log_qubit) 

    # Measure logical data from main qubit
    total_circuit.append("M",index_physical) 


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
    # We can use the target references, because we do the measurements in the correct order!
    def rel_meas(meas):
        return meas - total_circuit.num_measurements
    # the code space is defined by the stabilizer generators, which we measured using the ancilla qubits
    # the previous measurement defines the sign of the stabilizer generator! 

    offset_ancilla_0_X = 2*d*(d-1) 
    offset_ancilla_p_Z = 5*d*(d-1)

    offset_data_0 = 6*d*(d-1)
    offset_data_p = 6*d*(d-1) + (d**2 + (d-1)**2)

    targets_X, targets_Z = index_stab_targets(distance)
    # Detectors on |0> to detect Z-errors
    for i, targets in enumerate(targets_X):
        sign_defining_ancilla = i + offset_ancilla_0_X 
        recombined_stabilizer = [t + offset_data_0 for t in targets] 
        rel_meas_indices = [rel_meas(x) for x in [sign_defining_ancilla,*recombined_stabilizer]]
        total_circuit.append("DETECTOR",[stim.target_rec(x) for x in rel_meas_indices])

    # Detectors on |p> to detect X-errors
    for i, targets in enumerate(targets_Z):
        sign_defining_ancilla = i + offset_ancilla_p_Z 
        recombined_stabilizer = [t + offset_data_p for t in targets] 
        rel_meas_indices = [rel_meas(x) for x in [sign_defining_ancilla,*recombined_stabilizer]]
        total_circuit.append("DETECTOR",[stim.target_rec(x) for x in rel_meas_indices])

    #TODO construct detector that does a total measurement



    return total_circuit

diagram = generate_steane_set_up(distance=3).diagram("timeline-svg")
with open('new_test_circ.svg', 'w') as f:
    f.write(str(diagram))