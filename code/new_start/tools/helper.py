def save_circuit_diagram(circuit,savepath):
    diagram = circuit.diagram("timeline-svg")
    with open(savepath, 'w') as f:
        f.write(str(diagram))

# splits syndromes into parts
def split_syndrome(d,syndrome):
    """
    index of syndrome: i
    0       <= i <   d*(d-1): X-Stabilizer Syndrome
    d*(d-1) <= i < 2*d*(d-1): Z-Stabilizer Syndrome 
    """
    x_syndrome = syndrome[:d*(d-1)]
    z_syndrome = syndrome[-1*d*(d-1):]
    return x_syndrome, z_syndrome

def split_syndromes(d,syndromes):
    """
    index of syndrome: i
    0       <= i <   d*(d-1): X-Stabilizer Syndrome
    d*(d-1) <= i < 2*d*(d-1): Z-Stabilizer Syndrome 
    """
    x_syndromes = syndromes[:,:d*(d-1)]
    z_syndromes = syndromes[:,-1*d*(d-1):]
    return x_syndromes, z_syndromes
