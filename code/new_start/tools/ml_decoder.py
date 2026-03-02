import numpy as np
import scipy as sc

if __name__ == "__main__":
    from aron_ml import * 
    from helper import split_syndrome 
else:
    from tools.aron_ml import simulation_mld, error_converter
    from tools.helper import split_syndrome

## Syndrome to stabilizer Matrix

def x_syndrome_to_stabilizer_matrix(d, syndrome):    
    """
    from list to matrix shaped like the qubits location (d,d-1) 
    X stabilizer syndrome 
    """
    syndrome = np.multiply(syndrome,1) # Boolean to int
    stabilizer_matrix = np.zeros((d, d-1)) # final form 
    for col in range(d-1):
        for row in range(d):
            stabilizer_matrix[row,-1* (col + 1)] = syndrome[ col * d + row ]
    return stabilizer_matrix 

def z_syndrome_to_stabilizer_matrix(d, syndrome):
    """
    from list to matrix shaped like the qubits location (d,d-1) 
    Z stabilizer syndrome
    """
    syndrome = np.multiply(syndrome,1) # Boolean to int
    stabilizer_matrix = np.reshape(syndrome, (d, d-1))
    return stabilizer_matrix

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
    return f

if False:
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

    x_stab_m = x_syndrome_to_stabilizer_matrix(d,x_syn)
    x_f = stabilizer_to_pauli(d,x_stab_m)
    x_f_log = stabilizer_to_pauli(d,x_stab_m,add_logical=True)
    print(f"X-Stab. Matrix (rot.):\n{x_stab_m}")
    print(f"Z-Error Pauli:\n{x_f}")
    print_string(x_f)
    print(f"Z-Error Pauli + log.:\n{x_f_log}")
    print_string(x_f_log)
    print()

    z_stab_m = z_syndrome_to_stabilizer_matrix(d,z_syn)
    z_f = stabilizer_to_pauli(d,z_stab_m)
    z_f_log = stabilizer_to_pauli(d,z_stab_m,add_logical=True)
    print(f"Z-Stab. Matrix:\n{z_stab_m}")
    print(f"X-Error Pauli:\n{z_f}")
    print_string(z_f)
    print(f"X-Error Pauli + log.:\n{z_f_log}")
    print_string(z_f_log)
    print()

## ML Decoder 
# Matrix A
def gen_A(l):
    """
    return antisymetric matrix with l and -l on the diagonals

    :param l: numpy array representing a vector which is a element of R^m 
    """
    rows = [np.array(l), -1*np.array(l)]
    diags = [1,-1]
    a = sc.sparse.diags_array(rows, offsets = diags).toarray()
    return a

# Matrix D
def gen_D(l):
    """
    generate diagonal matrix 
    """
    d = sc.sparse.diags_array(l).toarray()
    return d

# Matrix M_0
def gen_m_0(d):
    l = (d-1)*[0,1] + [0] # no leading zero (due to being 1 offdiagonal)
    rows = [np.array(l), -1*np.array(l)]
    diags = [1,-1]
    m_0 = sc.sparse.diags_array(rows, offsets=diags, dtype=np.int64).toarray()
    m_0[0][-1] = 1
    m_0[-1][0] = -1
    return m_0

def calc_weights(p,f):
    weights = p**(1 - 2 * f) * (1 - p)**(-1 + 2 * f) 
    return weights

# normal probability 
def simulate_horizontal(d, j, m, gamma, weights):
    qubit_indices = j + (2 * d - 1) * np.arange(0,d)
    # subset of weights relevant 
    ws = weights[qubit_indices]
    # iterative
    for w in ws:
        gamma = gamma * (1 + w**2) / 2 
    # broadcasting
    t = (1 - ws**2) / (1 + ws**2) 
    s = (2 * ws) / (1 + ws**2) 
    # generation of A 
    v1 = np.zeros(2 * d - 1)
    v1[::2] = t
    a = gen_A(v1)
    # generation of B 
    v2 = np.repeat(s,2) 
    b = gen_D(v2)
    # Final calc:
    gamma  = gamma * np.sqrt(np.linalg.det(m + a))
    # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
    # Step 1: Solve (M + A) * X = B for X
    # x = np.linalg.solve(m + a, b)
    # Step 2: Compute B * X
    # m = a - b @ x @ b
    m = a - (b @ np.linalg.inv(m + a) @ b)
    return m, gamma 

def simulate_vertical(d, j, m, gamma, weights):
    qubit_indices = d + j + (2 * d - 1) * np.arange(d - 1)  
    # subset of weights relevant 
    ws = weights[qubit_indices]
    for w in ws:
        # iterative (TODO does not need to be iterative!)
        gamma = gamma * (1 + w**2)
    # safe some time by using numpy
    ts = (2 * ws) / (1 + ws**2) 
    ss = (1 - ws**2) / (1 + ws**2) 
    # Anti-symmetric matrix
    v1 = np.zeros(2 * d - 1) 
    v1[1::2] = ts
    a = gen_A(v1)
    # Diagonal matrix
    v2 = np.append(np.append([1],np.repeat(ss,2)),[1]) # not clean but = [1,s0,s0,...,sd-1,sd-1,1] 
    b = gen_D(v2)
    # Final calc:
    gamma  = gamma * np.sqrt(np.linalg.det(m + a))
    # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
    # Step 1: Solve (M + A) * X = B for X
    # x = np.linalg.solve(m + a, b)
    # Step 2: Compute B * X
    # m = a - b @ x
    m=a - (b @ np.linalg.inv(m + a) @ b)
    return m, gamma

def coset_probability(d,p,f):
    weights = calc_weights(p,f)
    m = gen_m_0(d)
    gamma = 2**(d - 1)

    # repr error prob
    n = d**2 + (d - 1)**2
    norm_f = np.sum(f)
    pauli_error_prob = (1 - p)**(n - norm_f) * p**norm_f
    for j in range(d - 1):
        m, gamma = simulate_horizontal(d, j, m, gamma, weights)
        m, gamma = simulate_vertical(d, j, m, gamma, weights)
    m, gamma = simulate_horizontal(d, d-1, m, gamma, weights) # d-1 due to 0 <= i < d and not 1<=i<=d
    coset_prob = pauli_error_prob * np.sqrt(gamma / 2) * (np.linalg.det((m + gen_m_0(d))))**(1/4)
    return coset_prob 

def decode_half_syndrome(d, p, h_syndrome, stab_type="Z",only_obs_flip=True):
    if stab_type.upper() == "Z":
        stabilizer_matrix = z_syndrome_to_stabilizer_matrix(d, h_syndrome)
    elif stab_type.upper() == "X": #TODO check this!
        stabilizer_matrix = x_syndrome_to_stabilizer_matrix(d, h_syndrome)
    else:
        raise ValueError("unexpected detector")
    
    # prob of coset without logical error 
    f = stabilizer_to_pauli(d, stabilizer_matrix)
    p_I = coset_probability(d, p, f)
    # prob of coset with logical error 
    f = stabilizer_to_pauli(d, stabilizer_matrix, add_logical=True)
    p_L = coset_probability(d, p, f)

    obs_flip = True if p_I < p_L else False
    if only_obs_flip:
        return obs_flip
    else: 
        return p_I, p_L, obs_flip 

# log prob
def simulate_horizontal_log(d, j, m, log_gamma, weights):
    qubit_indices = j + (2 * d - 1) * np.arange(0,d)
    # subset of weights relevant 
    ws = weights[qubit_indices]
    # iterative
    log_gamma += np.sum(np.log((1+ws**2)/2))
    # broadcasting
    t = (1 - ws**2) / (1 + ws**2) 
    s = (2 * ws) / (1 + ws**2) 
    # generation of A 
    v1 = np.zeros(2 * d - 1)
    v1[::2] = t
    a = gen_A(v1)
    # generation of B 
    v2 = np.repeat(s,2) 
    b = gen_D(v2)
    # Final calc:
    log_gamma += np.log(np.sqrt(np.linalg.det(m + a)))
    # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
    # Step 1: Solve (M + A) * X = B for X
    # x = np.linalg.solve(m + a, b)
    # Step 2: Compute B * X
    # m = a - b @ x @ b
    m = a - (b @ np.linalg.inv(m + a) @ b)
    return m, log_gamma

def simulate_vertical_log(d, j, m, log_gamma, weights):
    qubit_indices = d + j + (2 * d - 1) * np.arange(d - 1)  
    ws = weights[qubit_indices]
    log_gamma += np.sum(np.log(1+ws**2))
    ts = (2 * ws) / (1 + ws**2) 
    ss = (1 - ws**2) / (1 + ws**2) 
    # Anti-symmetric matrix
    v1 = np.zeros(2 * d - 1) 
    v1[1::2] = ts
    a = gen_A(v1)
    # Diagonal matrix
    v2 = np.append(np.append([1],np.repeat(ss,2)),[1]) # not clean but = [1,s0,s0,...,sd-1,sd-1,1] 
    b = gen_D(v2)
    # Final calc:
    log_gamma += np.log(np.sqrt(np.linalg.det(m + a)))
    # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
    # Step 1: Solve (M + A) * X = B for X
    # x = np.linalg.solve(m + a, b)
    # Step 2: Compute B * X
    # m = a - b @ x
    m=a - (b @ np.linalg.inv(m + a) @ b)
    return m, log_gamma

def coset_probability_log(d,p,f):
    weights = calc_weights(p,f)

    m = gen_m_0(d)
    log_gamma = np.log(2**(d-1))

    # repr error prob
    n = d**2 + (d - 1)**2
    norm_f = np.sum(f)
    pauli_error_prob = (1 - p)**(n - norm_f) * p**norm_f

    for j in range(d - 1):
        m, log_gamma = simulate_horizontal_log(d, j, m, log_gamma, weights)
        m, log_gamma = simulate_vertical_log(d, j, m, log_gamma, weights)
    m, log_gamma = simulate_horizontal_log(d, d-1, m, log_gamma, weights) # d-1 due to 0 <= i < d and not 1<=i<=d
    log_coset_prob = 1/2 * log_gamma - 1/2 * np.log(2) + np.log(pauli_error_prob)  + 1/4*np.log(np.linalg.det(m + gen_m_0(d)))
    return log_coset_prob

def decode_half_syndrome_log(d, p, h_syndrome, stab_type="Z",only_obs_flip=True):
    if stab_type.upper() == "Z":
        stabilizer_matrix = z_syndrome_to_stabilizer_matrix(d, h_syndrome)
    elif stab_type.upper() == "X":
        stabilizer_matrix = x_syndrome_to_stabilizer_matrix(d, h_syndrome)
    else:
        raise ValueError("unexpected detector")
    
    # prob of coset without logical error 
    f = stabilizer_to_pauli(d, stabilizer_matrix)
    log_p_I = coset_probability_log(d, p, f)
    # prob of coset with logical error 
    f = stabilizer_to_pauli(d, stabilizer_matrix, add_logical=True)
    log_p_L = coset_probability_log(d, p, f)

    obs_flip = True if log_p_I < log_p_L else False

    if only_obs_flip:
        # returns if observable flips
        return obs_flip
    else: 
        return log_p_I, log_p_L, obs_flip 

# arons code adapted 
def combined_aron(d,p,h_syndrome, only_obs_flip=True):
    z_matrix = z_syndrome_to_stabilizer_matrix(d, h_syndrome)
    f_I = stabilizer_to_pauli(d, z_matrix)
    f_L = stabilizer_to_pauli(d, z_matrix, add_logical=True)
    p_I = simulation_mld(
        p,
        d,
        error_converter(int(d), f_I),
        ) 
    p_L = simulation_mld(
        p,
        d,
        error_converter(int(d), f_L),
        )
    obs_flip = True if p_I < p_L else False
    if only_obs_flip:
        return obs_flip
    else:
        return p_I, p_L, obs_flip

if __name__ == "__main__" and False:
    # compare arons and my decoder 
    def check_prop_array(pis,pls,log_values=False):
        pis = np.array(pis)
        pls = np.array(pls)
        if log_values:
            pis = np.exp(pis)
            pls = np.exp(pls)
        print(f"\nsum p_I={np.sum(pis)}")
        print(f"sum p_L={np.sum(pls)}")
        print(f"Total sum={np.sum(pls) + np.sum(pis)}")

    def test(d):
        """
        generates all possible syndromes 
        """
        ns = np.arange(0,2**(d),)
        occ_vec = ((ns[:, None] >> np.arange(d)) & 1).astype(np.bool) 
        return occ_vec



    syndrome_list3 = [
        True, False,
        True, False,
        True, True,
    ]
    syndrome_list = syndrome_list3 
    stab_type = "Z"
    d = 3
    p = 0.01 
    all_possible_syndromes = test(d*(d-1))
    p_is = [] 
    ap_is = []
    p_ls = []
    ap_ls = []
    obs_flips = []
    aobs_flips = []
    miss_count = 0
    my_log_implementation = False# for log = False
    for synd in all_possible_syndromes:
        if not my_log_implementation:
            p_I, p_L, obs = decode_half_syndrome(d, p, synd, stab_type, only_obs_flip=False)
        else:
            p_I, p_L, obs = decode_half_syndrome_log(d, p, synd, stab_type, only_obs_flip=False)
        p_is += [p_I]
        p_ls += [p_L]
        obs_flips += [obs]
        ap_I, ap_L, aobs = combined_aron(d,p,synd) # always in log scale
        ap_is += [ap_I]
        ap_ls += [ap_L]
        aobs_flips += [aobs]
        if obs != aobs:
            miss_count += 1
            print()
            print(obs,aobs)
            print(synd)
            print(p_I, p_L)
            print(ap_I, ap_L)
            print()


    print("different obs:",miss_count)
    print(len(all_possible_syndromes))
    print("mine:") 
    check_prop_array(p_is, p_ls, log_values=my_log_implementation)
    # = 0.999999999 for normal
    # = 0.999999999 for log
    print("aron:")
    check_prop_array(ap_is, ap_ls, log_values=True)
    # works now!