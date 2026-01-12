import numpy as np 
import stim


def index_qubits_surface_code(distance: int = 3, offset: int = 0):
    if distance%2!=1:
        raise ValueError(f"Odd number expected, got {distance}")
    n_total = 0
    # physical qubits: (horizontal) + (vertical) edges
    n_physical = distance**2 + (distance-1)**2
    index_physical = np.arange(n_total,n_total+n_physical) + offset
    n_total += n_physical
    # plaquette stabilizer (X)
    n_X_stab = distance * (distance-1) 
    index_X_stab = np.arange(n_total,n_total+n_X_stab) + offset
    n_total += n_X_stab 
    # site stabilizer (Z)
    n_Z_stab = distance * (distance-1) 
    index_Z_stab = np.arange(n_total,n_total+n_Z_stab) + offset
    n_total += n_Z_stab 

    return index_physical, index_X_stab, index_Z_stab 

def index_stab_targets(distance: int = 3, offset: int = 0):
    if distance%2!=1:
        raise ValueError(f"Odd number expected, got {distance}")
    d = distance
    targets_X_stab, targets_Z_stab = [],[] 
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
            targets_X_stab.append(targets)
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
            targets_Z_stab.append(targets)
    return targets_X_stab, targets_Z_stab

def generate_surface_code_circuit(distance: int = 3, offset=0, state: str ="0"):
    circuit = stim.Circuit()

    index_physical, index_X_stab, index_Z_stab = index_qubits_surface_code(distance, offset)
    # init qubits by resetting them to 0
    if state == "0": # |0>
        circuit.append("R", index_physical, tag="init")
    elif state == "p": # |+>
        circuit.append("RX", index_physical, tag="init")
    else:
        raise ValueError(f"does not know state {state}, expected: '0' for |0> or 'p' for |+>")
    circuit.append("R", index_X_stab, tag="init")
    circuit.append("R", index_Z_stab, tag="init")

    # Init into a logical state by measurement of stabilizers
    targets_X_stab, targets_Z_stab = index_stab_targets(distance, offset)
    # Plaquette-/X-stabilizers
    circuit.append("H", index_X_stab)
    for i_ancilla ,targets in zip(index_X_stab,targets_X_stab):
        for i_target in targets:
            circuit.append("CNOT", [i_ancilla, i_target])
    circuit.append("H", index_X_stab)
    circuit.append("MR", index_X_stab) # <- redefines the codespace/generators of stabilizers! 
    # Site-/Z-stabilizers
    for i_ancilla ,targets in zip(index_Z_stab,targets_Z_stab):
        for i_target in targets:
            circuit.append("CNOT", [i_target, i_ancilla])
    circuit.append("MR", index_Z_stab) # <- redefines the codespace/generators of stabilizers!
    return circuit

def generate_steane_set_up(distance: int = 3):
    d = distance
    offset_per_log_qubit = (d**2 + (d-1)**2 + 2*d*(d-1))
    # init all qubits!
    circuit_log_data = generate_surface_code_circuit(distance, offset=0)
    circuit_aux_0_log = generate_surface_code_circuit(
        distance, 
        offset = offset_per_log_qubit,
        )
    circuit_aux_p_log = generate_surface_code_circuit(
        distance, 
        offset = 2*offset_per_log_qubit,
        state="p",
        )
    total_circuit = circuit_log_data + circuit_aux_0_log + circuit_aux_p_log

    # Steane Magic! 
    index_physical, index_X_stab, index_Z_stab = index_qubits_surface_code(distance)
    # entangle C|0>NOT|Psi> 
    for i in index_physical:
        # CxNOTy: "CNOT",[x,y]
        total_circuit.append("CNOT", (offset_per_log_qubit + i, i))
    # measure:
    total_circuit.append("H", index_physical + offset_per_log_qubit) 
    # note the Hadamard is NOT! a log. Hadamard (not transversal)
    total_circuit.append("MR",index_physical + offset_per_log_qubit) 
    # Recombine into stabilizer using index_X_stab

    # entangle C|Psi>NOT|p> 
    for i in index_physical:
        # CxNOTy: "CNOT",[x,y]
        total_circuit.append("CNOT", (i, 2*offset_per_log_qubit + i))
    # measure:
    total_circuit.append("MR",index_physical + 2*offset_per_log_qubit) 
    # Recombine into stabilizer using index_Z_stab

    # Total Measurement needed?!

    return total_circuit

diagram = generate_steane_set_up(distance=3).diagram("timeline-svg")
with open('new_test_circ.svg', 'w') as f:
    f.write(str(diagram))