from tools.surface_code import generate_steane_circuit
from tools.error_models import add_noise
noise = 0.1
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
basic_noise_model = {
    "initialize": {
        "operator": {"I"}, # Trick I use because init algo is out of scope! 
        "error_position": "following",
        "error": "X_ERROR",
        "noise": 0.003,
        "specific_tag": {"psi_data",}, # specific tag of targeted instruction
    },
}

noiseless_circuit = generate_steane_circuit(distance=3)
noisy_circuit = add_noise(noiseless_circuit, noise_model)

diagram = noisy_circuit.diagram("timeline-svg")
with open('new_test_circ.svg', 'w') as f:
    f.write(str(diagram))