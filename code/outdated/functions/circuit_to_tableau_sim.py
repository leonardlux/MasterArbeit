"""Module converting stim.Circuit into stim.TableuSimulator"""
import stim

# constants with relevant names (sets) 
ALL_GATES = {
    "C_XYZ", "C_ZYX", "H", "H_YZ", "I","SQRT_Y","SQRT_Y_DAG", 
    "CX", "CY","CZ","XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ",  
    "RX", "RY","M", "MX", "MY","X","Y","Z", "MPP", "R"}
SINGLE_QUBIT_GATES = {
    "H","I","SQRT_Y","SQRT_Y_DAG", "R", "RX", "RY","M", "MX",
    "MY","X","Y","Z"
    }
ALL_ERROS = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR","PAULI_CHANNEL_1"}
ANNOTATION_OPS = {"OBSERVABLE_INCLUDE", "DETECTOR", "SHIFT_COORDS", "QUBIT_COORDS"}

"""
Basic idea:
iterating over my finished ciruit to turn it into a tableau
"""

def apply_gates(name: str,
                targets: list,
                simulator: stim.TableauSimulator) -> None:
    """
    add gate(s) with given name and targets to given simulator 
    """
    # one gate can have multiple targets => multiple gates
    if name == 'H':
        simulator.h(*targets)
    elif name == 'S':
        simulator.s(*targets)
    elif name == 'X':
        simulator.x(*targets)
    elif name == 'Y':
        simulator.y(*targets)
    elif name == 'Z':
        simulator.z(*targets)
    elif name == 'SQRT_Y':
        simulator.sqrt_y(*targets)
    elif name == 'SQRT_Y_DAG':
        simulator.sqrt_y_dag(*targets)
    elif name == 'CNOT':
        simulator.cnot(*targets)
    elif name == 'CX':
        simulator.cnot(*targets)
    elif name == 'CZ':
        simulator.cz(*targets) #standard gate
    elif name == 'SWAP':
        simulator.swap(*targets)
    elif name == 'M':
        for target in targets:
            simulator.measure(target.value)
    elif name == 'MX':
        # first apply hadamard and then measure
        simulator.h(*targets)
        for target in targets:
            simulator.measure(target.value)
    elif name == 'R':
        simulator.reset(*targets)
    elif name == 'MPP':
        the_circuit = stim.Circuit()
        the_circuit.append("MPP", targets)
        simulator.do_circuit(the_circuit)
    elif name == 'TICK':
        #FIXME TICK is not a gate
        pass
    else:
        raise ValueError(f"Unsupported gate: {name}")


def apply_error(name,
                probabilities,
                targets,
                simulator) -> None:
    """
    add errors with given name, prob and targets to simulator 
    """
    if name == 'DEPOLARIZE1':
        simulator.depolarize1(*targets,p=probabilities[0])
    elif name == 'DEPOLARIZE2':
        simulator.depolarize2(*targets,p=probabilities[0])
    elif name == 'X_ERROR':
        simulator.x_error(*targets,p=probabilities[0])
    elif name == 'Z_ERROR':
        simulator.z_error(*targets,p=probabilities[0])
    elif name == 'PAULI_CHANNEL_1':
        the_circuit = stim.Circuit()
        the_circuit.append("PAULI_CHANNEL_1", *targets,probabilities)
        simulator.do_circuit(the_circuit)
    elif name == 'PAULI_CHANNEL_2':
        the_circuit = stim.Circuit()
        the_circuit.append("PAULI_CHANNEL_2", *targets,probabilities)
        simulator.do_circuit(the_circuit)
    else:
        raise ValueError(f"Unsupported gate: {name}")


# Understand this function!
def apply_annotation(name: str,
                     targets: list,
                     simulator: stim.TableauSimulator):
    """
    Extract the relevant information from the annotation and then add relevant parts
    to the simulator.
    """
    measurements = simulator.current_measurement_record()
    if name == 'DETECTOR':
        detector = 0
        for x in targets:
            detector += int(measurements[x.value])
        return detector%2
 
    elif name == 'OBSERVABLE_INCLUDE':
        obs = 0
        for x in targets:
            obs +=  int(measurements[x.value])
        return obs%2
 
    elif name == 'QUBIT_COORDS':
        pass
 
    elif name == 'SHIFT_COORDS':
        pass
 
    else:
        print(name)
        raise ValueError(f"Unsupported gate: {name}")
 

def circuit_to_tableau_simulator(circuit: stim.Circuit, num_qubits):

    """
    Run a noisy stim circuit as a tableu simulator
    return list of detection events and observables 
    
    """
 
    simulator = stim.TableauSimulator()
    simulator.set_num_qubits(num_qubits)
 
    detectors = []
    observables = []
 
    for operation in circuit:
        gate = operation.name  # Access the gate type
        targets = operation.targets_copy()  # Access the target qubits
        arguments = operation.gate_args_copy() # Access arguments of the operation
 
        #print(gate,targets,arguments)
 
        if gate in all_gates: #gate
            apply_gates(gate,targets,simulator)
 
            #previous_gate = gate #store the last gate/measurement operation
 
            #no_idling_qubits = []
            #for tg in targets:
            #    no_idling_qubits.append(tg.value) #store the qubits that were touche
 
        elif gate in all_errors: #error
            apply_error(gate,arguments,targets,simulator)
            
        elif gate in ANNOTATION_OPS:
            detect_event = apply_annotation(gate,targets,simulator)
            if gate == 'DETECTOR': detectors.append(detect_event)
            if gate == 'OBSERVABLE_INCLUDE': observables.append(detect_event)
            
            #FIXME: coordinates of the stabilizers are ignored
            
            
        elif gate == 'TICK': #Apply leakage decay
            
            pass
            #print(unleak)
   
                    
        else: print("something is missing " + gate)
            
        
    return detectors, simulator
