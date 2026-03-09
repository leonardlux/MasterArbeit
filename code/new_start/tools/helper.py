import numpy as np

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

# Everything following might be broken!
def split_syndromes_rounds(d, rounds, syndromes):
    """
    n_stab = d*(d-1)

    index of syndrome: 
    start of new round: offset = r * 2 * n_stab 
        0*n_stab <= i < 1*n_stab: X-Stabilizer Syndrome
        1*n_stab <= i < 2*n_stab: Z-Stabilizer Syndrome 
    end of round 
        reps * 2 * n_stab <= i < (2 * reps + 1) * n_stab: Z-Stabilizer Syndrome for Fault Tolerance
    """
    n_stab = d*(d-1)
    n_round = 2*n_stab

    x_syndromes  = [syndromes[:, r*n_round           : r*n_round +   n_stab] for r in range(rounds)]
    z_syndromes  = [syndromes[:, r*n_round + n_stab  : r*n_round + 2*n_stab] for r in range(rounds)]
    ft_syndromes = syndromes[:,-1*n_stab:]
    """
    shape of return:
    x_syndromes[round][shot][stab] 
    """
    return x_syndromes, z_syndromes, ft_syndromes

def xor_syndromes(rounds, syndromes):
    # idea: xor all the syndrome with one another to track the acutal change
    xor_syndromes = np.zeros(np.array(syndromes).shape)
    xor_syndromes[0] = syndromes[0]
    for round in range(rounds-1):
        xor_syndromes[round+1] = np.logical_xor(syndromes[round],syndromes[round+1])
    return xor_syndromes

def xor_ft_syndrome(ft_synd, stab_synd):
    xor_ft_synd = np.logical_xor(stab_synd, ft_synd)
    return xor_ft_synd 

def split_and_xor_syndrome(d, rounds, syndromes, ft_z_stab=True):
    """
    reps: number of repetions 
    syndrome: output of stim 
    z_stab: determines which synd is used to construct pauli frame FT
        -> True <=> z_stab
        -> False <=> x_stab
    this function does necessary all the preparations for the syndromes 
    """
    x_syndromes, z_syndromes, ft_syndromes = split_syndromes_rounds(d, rounds, syndromes)
    # pauli frame:
    px_synd  = xor_syndromes(rounds,x_syndromes) 
    pz_synd  = xor_syndromes(rounds,z_syndromes) 
    if ft_z_stab:
        # observable is measured in Z basis
        pft_synt = xor_ft_syndrome(z_syndromes[-1],ft_syndromes)
    elif not ft_z_stab:
        # observable is measured in X basis
        pft_synt = xor_ft_syndrome(x_syndromes[-1],ft_syndromes)
    else:
        raise ValueError("Unkown state")
    # Order of snydromes synd[round][shot][stab]
    return px_synd, pz_synd, pft_synt

def reorder_syndromes(old_order):
    new_order = old_order.transpose(1, 0, 2)
    return new_order