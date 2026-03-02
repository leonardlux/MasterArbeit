def save_circuit_diagram(circuit,savepath):
    diagram = circuit.diagram("timeline-svg")
    with open(savepath, 'w') as f:
        f.write(str(diagram))

# splits syndromes into parts
def split_syndrome(d,syndrome):
    """
    index of syndrome: i
    0         <= i <   d*(d-1): X-Stabilizer Syndrome
    1*d*(d-1) <= i < 2*d*(d-1): Z-Stabilizer Syndrome 
    2*d*(d-1) <= i < 3*d*(d-1): Z-Stabilizer Syndrome for Fault Tolerance
    """
    n_stab = d*(d-1)
    x_syndrome = syndrome[        :  n_stab]
    z_syndrome = syndrome[  n_stab:2*n_stab]
    return x_syndrome, z_syndrome

def split_syndromes(d,syndromes):
    """
    index of syndrome: i
    0         <= i <   d*(d-1): X-Stabilizer Syndrome
    1*d*(d-1) <= i < 2*d*(d-1): Z-Stabilizer Syndrome 
    2*d*(d-1) <= i < 3*d*(d-1): Z-Stabilizer Syndrome for Fault Tolerance
    """
    n_stab = d*(d-1)
    x_syndromes  = syndromes[:,        :  n_stab]
    z_syndromes  = syndromes[:,  n_stab:2*n_stab]
    ft_syndromes = syndromes[:,2*n_stab:3*n_stab]

    return x_syndromes, z_syndromes, ft_syndromes
