# This file contains old code that was cut, but might be useful (to lazy to search git history)

# Function to generate the coords to have a layout string that resembles the usual surface 17 code 
def gen_layout_string(x_offset=0,y_offset=0):
    # min offset to have them next to another is 6 in each direction
    data_coords = [ (x,y) for y in [1,3,5] for x in [1,3,5] ]
    Z_qubits_coords = [
        (0,2),
        (4,2),
        (2,4),
        (6,4),
        ]
    X_qubits_coords = [
        (4,0),
        (2,2),
        (4,4),
        (2,6),
    ]
    data_coords = coords_offset(data_coords,x_offset,y_offset)
    Z_qubits_coords = coords_offset(Z_qubits_coords,x_offset,y_offset)
    X_qubits_coords = coords_offset(X_qubits_coords,x_offset,y_offset)
    return data_coords, Z_qubits_coords, X_qubits_coords 

def coords_offset(coords, x_offset, y_offset,):
    return [(coord[0]+x_offset, coord[1]+y_offset)for coord in coords]

## Code adds coordinates to my qubits in the process of initilizsation
    circuit.append_from_stim_program_text("SHIFT_COORDS(7, 0, 1)") # shifts each set of qubits next to another
    for i,qubit_idx in enumerate(data_qubits):
        x = data_coords[i][0]
        y = data_coords[i][1]
        circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x,y) + " {}".format(qubit_idx))
    # same for stabilizers
        for i,qubit_idx in enumerate(Z_measure_qubits):
            x = X_qubits_coords[i][0]
            y = X_qubits_coords[i][1]
            circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x,y) + " {}".format(qubit_idx))
        for i,qubit_idx in enumerate(X_measure_qubits):
            x = Z_qubits_coords[i][0]
            y = Z_qubits_coords[i][1]
            circuit.append_from_stim_program_text("QUBIT_COORDS({},{})".format(x,y) + " {}".format(qubit_idx))

# Debugging Tool

def undetectable_log_errors(circuit):
    ule =  circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=10,
        dont_explore_edges_with_degree_above=10,
        dont_explore_edges_increasing_symptom_degree=False, 
        canonicalize_circuit_errors=True
        )
    for u in ule: 
        print(u)
        # print("location: ",u.circuit_error_locations) 
        # print("dem error terms: ",u.dem_error_terms)
        print("length of undetectable logical error = ", len(ule))
        print()

### Example of a noisy code
steane_circuit = construct_steane_circuit()

noisy_circuit = add_noise(
    steane_circuit,
    noise_model,
    single_qubit_gate_errors=False,
    measurement_errors=False,
    initialize_errors=False,
    )
if 1:
    diagram = noisy_circuit.diagram("timeline-svg")  
    with open('noisy_circuit.svg', 'w') as f:
        f.write(str(diagram))

    diagram = noisy_circuit.diagram("timeslice-svg")  
    with open('timeslice_noisy.svg', 'w') as f:
        f.write(str(diagram))
    
### Example of a different noisy code
reversed_steane_circuit = construct_steane_circuit(reverse_order=True)

noisy_reversed_circuit = add_noise(
    reversed_steane_circuit,
    noise_model,
    single_qubit_gate_errors=False,
    measurement_errors=False,
    initialize_errors=False,
    )
if 1:
    diagram = noisy_circuit.diagram("timeline-svg")  
    with open('noisy_reversed_circuit.svg', 'w') as f:
        f.write(str(diagram))


### Detection Error model
if 0:
    dem = noisy_circuit.detector_error_model()
    print(repr(dem))
    dem.diagram("matchgraph-svg")
    diagram = dem.diagram("matchgraph-svg")
    with open('matchgrapg.svg', 'w') as f:
        f.write(str(diagram))