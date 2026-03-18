import numpy as np

if __name__ == "__main__":
    from syndrome import split_syndrome 
    from surface_code import index_log_Z, index_log_X
else:
    from tools.syndrome import split_syndrome
    from tools.surface_code import index_log_Z, index_log_X

## Syndrome to stabilizer Matrix

def format_syndrome_to_matrix(d, syndrome):
    """
    from list to matrix shaped like the qubits location (d,d-1) 
    """
    syndrome = np.multiply(syndrome,1) # Boolean to int
    stabilizer_matrix = np.reshape(syndrome, (d, d-1))
    return stabilizer_matrix

def rotate_and_reorder_syndrome(d, syndrome):    
    """
    rotate syndrome: rotate stabilizer position 90° counterclockwise
    then reorder syndromes to have the same ordering (positions)
    """
    syndrome = np.multiply(syndrome,1) # Boolean to int
    rot_syndrome = np.zeros((d*(d-1))) # final form 
    # first map the rotated stabilizer to the corresponding index of unrotated stabilize index of unrotated stabilizerr 
    for i_row in range(d):
        for i_col in range(d-1):
            rot_syndrome[i_col + i_row * (d-1)] = syndrome[(d-(i_row+1)) + d * i_col]
    return rot_syndrome 

## Stabilizer matrix to Pauli error 
def stabilizer_to_pauli(d, syndrome_matrix, add_logical: bool = False):
    """
    given a matrix of syndrome calculates a possible X-error 

    syndrome array: indedx: starting top left, going right -> down
    """
    # at this step only errors on horizontal qubits are generated 
    f = np.zeros((d,d)) 
    for i, row in enumerate(syndrome_matrix):
        for j, detector in enumerate(row):
            if detector:
                for jt in range(j+1):
                    f[i,jt] = (f[i,jt] + 1)%2
    # add logical operator by inverting first row (applying a logical gate) 
    if add_logical:
        f[0,:] = (f[0,:] + 1) % 2  
    # add the vertical qubits erros to pauli string in 0 state (no error)
    f = np.concatenate((f,np.zeros((d,d-1))),axis=1)
    # flatten so that we can refer to them by index of qubit/edge location!
    f = f.flatten()
    f = f[:-(d-1)] # last row of verticals does not exists!

    # determine p for Z * f = p * f * Z
    # determines if log, measured observable should be flipped to account for coset pauli
    pauli_flip = True if np.sum(f[index_log_Z(d)])%2 == 1 else False
    return f, pauli_flip 


## Combined:
def syndrome_to_pauli_flips(d, syndrome):
    # Split-Syndrome
    x_syn, z_syn = split_syndrome(d,syndrome)

    x_stab_m = rotate_and_reorder_syndrome(d,x_syn)
    _, log_z_flip_pauli = stabilizer_to_pauli(d,x_stab_m)

    z_stab_m = format_syndrome_to_matrix(d,z_syn)
    _, log_x_flip_pauli = stabilizer_to_pauli(d,z_stab_m)
    # returns for both log obsvervable if they would flip due to the reference pauli
    return log_x_flip_pauli, log_z_flip_pauli

if  __name__ == "__main__" and False:
    # check function of syndrome -> pauli functions 
    d = 3 
    flipped_bits = [-1]

    def print_string(f):
        print("Error as Matrix:")
        for i in range(d-1):
            print(f[i*(2*d-1):i*(2*d-1)+d])
            print(f[i*(2*d-1)+d:i*(2*d-1)+d+(d-1)])
        print(f[-d:])

    syndrome = [False] * (2*d*(d-1))
    for flip in flipped_bits:
        syndrome[flip] = not syndrome[flip]

    print(f"complete syndrome: \n{syndrome}\n")

    x_syn, z_syn = split_syndrome(d,syndrome)

    print(f"syndrome parts: \nX-Stab.: {x_syn} \nZ-Stab.: {z_syn}\n")

    x_stab_m = rotate_and_reorder_syndrome(d,x_syn)
    x_f, sgn_f = stabilizer_to_pauli(d,x_stab_m)
    x_f_log, _ = stabilizer_to_pauli(d,x_stab_m,add_logical=True)
    print(f"X-Stab. Matrix (rot.):\n{x_stab_m}")
    print(f"Z-Error Pauli:\n{x_f}")
    print(f"sign: {sgn_f}")
    print_string(x_f)
    print(f"Z-Error Pauli + log.:\n{x_f_log}")
    print_string(x_f_log)
    print()

    z_stab_m = format_syndrome_to_matrix(d,z_syn)
    z_f, sgn_f = stabilizer_to_pauli(d,z_stab_m)
    z_f_log, _ = stabilizer_to_pauli(d,z_stab_m,add_logical=True)
    print(f"Z-Stab. Matrix:\n{z_stab_m}")
    print(f"X-Error Pauli:\n{z_f}")
    print(f"sign: {sgn_f}")
    print_string(z_f)
    print(f"X-Error Pauli + log.:\n{z_f_log}")
    print_string(z_f_log)
    print()

