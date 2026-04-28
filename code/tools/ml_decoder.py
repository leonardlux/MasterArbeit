import numpy as np
import scipy as sc
import numba

if __name__ == "__main__":
    from aron_ml import * 
    from pauli_frame_track import stabilizer_to_pauli, rotate_and_reorder_syndrome, format_syndrome_to_matrix
else:
    from tools.aron_ml import simulation_mld, error_converter
    from tools.pauli_frame_track import stabilizer_to_pauli, rotate_and_reorder_syndrome, format_syndrome_to_matrix

## ML Decoder 
# Matrix M_0
@numba.njit
def gen_m0(d, dtype=np.float32):
    m0=np.zeros((2*d,2*d), dtype=dtype)
    for i in range(d-1):
        m0[2*i+1,2*i+2]=1
        m0[2*i+2,2*i+1]=-1

    m0[0,2*d-1]=1
    m0[2*d-1,0]=-1
    return m0

@numba.njit
def calc_weights(p,f):
    weights = p**(1 - 2 * f) * (1 - p)**(-1 + 2 * f) 
    return weights

# normal probability 
@numba.njit
def simulate_horizontal(d, j, m, gamma, weights, dtype=np.float32):
    A = np.zeros((2*d,2*d), dtype=dtype)
    B = np.zeros((2*d,2*d), dtype=dtype)

    # gather correct weights 
    for i in range(d):
        qubit_index = j + (2 * d - 1) * i
        w = weights[qubit_index] 
        gamma = gamma * (1 + w**2) / 2
        t = (1 - w**2) / (1 + w**2) 
        s = (2 * w)    / (1 + w**2) 
        A[2*i,   2*i+1] = t
        A[2*i+1, 2*i  ] =-t
        B[2*i,   2*i  ] = s
        B[2*i+1, 2*i+1] = s
    
    gamma = gamma * np.sqrt(np.linalg.det(m + A))
    # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
    # Step 1: Solve (M + A) * X = B for X
    # x = np.linalg.solve(m + A, B)
    # Step 2: Compute B * X
    # m = A - B @ x 
    m = A - (B @ np.linalg.inv(m + A) @ B)
    return m, gamma

@numba.njit
def simulate_vertical(d, j, m, gamma, weights, dtype=np.float32):
    A = np.zeros((2*d,2*d), dtype=dtype)
    B = np.zeros((2*d,2*d), dtype=dtype)
    B[0,0] = 1
    B[2*d-1,2*d-1]=1

    for i in range(d-1):
        qubit_index = d + j + (2 * d - 1) * i 
        w = weights[qubit_index] 
        gamma = gamma * (1 + w**2)
        t = (2 * w)    / (1 + w**2) 
        s = (1 - w**2) / (1 + w**2)
        A[2*i+1, 2*i+2] = t
        A[2*i+2, 2*i+1] =-t
        B[2*i+1, 2*i+1] = s
        B[2*i+2, 2*i+2] = s
    gamma  = gamma * np.sqrt(np.linalg.det(m + A))
    # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
    # Step 1: Solve (M + A) * X = B for X
    # x = np.linalg.solve(m + A, B)
    # Step 2: Compute B * X
    # m = A - B @ x
    m = A - (B @ np.linalg.inv(m + A) @ B)
    return m, gamma

@numba.njit
def coset_probability(d,p,f, dtype=np.float32):
    weights = calc_weights(p,f)
    m = gen_m0(d)
    gamma = 2**(d - 1)
    for j in range(d - 1):
        m, gamma = simulate_horizontal(d, j, m, gamma, weights, dtype=dtype)
        m, gamma = simulate_vertical(d, j, m, gamma, weights, dtype=dtype)
    m, gamma = simulate_horizontal(d, d-1, m, gamma, weights, dtype=dtype) # d-1 due to 0 <= i < d and not 1<=i<=d

    # repr error prob
    n = d**2 + (d - 1)**2
    norm_f = np.sum(f)
    pauli_error_prob = (1 - p)**(n - norm_f) * p**norm_f

    # calc coset
    coset_prob = pauli_error_prob * np.sqrt(gamma / 2) * (np.linalg.det((m + gen_m0(d))))**(1/4)
    return coset_prob 

@numba.njit
def decode_half_syndrome(d, p, h_syndrome, stab_type="Z",dtype=np.float32):
    if stab_type.upper() == "Z":
        stabilizer_matrix = format_syndrome_to_matrix(d, h_syndrome)
    elif stab_type.upper() == "X": 
        rot_synd = rotate_and_reorder_syndrome(d, h_syndrome)
        stabilizer_matrix = format_syndrome_to_matrix(d, rot_synd)
    else:
        raise ValueError("unexpected detector")
    
    # prob of coset without logical error 
    f, c_f = stabilizer_to_pauli(d, stabilizer_matrix)
    p_I = coset_probability(d, p, f, dtype=dtype)
    # prob of coset with logical error 
    f, _ = stabilizer_to_pauli(d, stabilizer_matrix, add_logical=True)
    p_L = coset_probability(d, p, f, dtype=dtype)
    obs_flip = True if p_I < p_L else False
    # c_f is the sign if we commute the logical with the pauli
    return obs_flip, c_f

# log prob
@numba.njit
def simulate_horizontal_log(d, j, m, log_gamma, weights, dtype=np.float32):
    A = np.zeros((2*d,2*d), dtype=dtype)
    B = np.zeros((2*d,2*d), dtype=dtype)

    # gather correct weights 
    for i in range(d):
        qubit_index = j + (2 * d - 1) * i
        w = weights[qubit_index] 
        log_gamma = log_gamma + np.log((1 + w**2) / 2)
        t = (1 - w**2) / (1 + w**2) 
        s = (2 * w)    / (1 + w**2) 
        A[2*i,   2*i+1] = t
        A[2*i+1, 2*i  ] =-t
        B[2*i,   2*i  ] = s
        B[2*i+1, 2*i+1] = s
    
    gamma = gamma + np.log(np.sqrt(np.linalg.det(m + A)))
    # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
    # Step 1: Solve (M + A) * X = B for X
    x = np.linalg.solve(m + A, B)
    # Step 2: Compute B * X
    m = A - B @ x 
    # m = a - (b @ np.linalg.inv(m + a) @ b)
    return m, log_gamma

@numba.njit
def simulate_vertical_log(d, j, m, log_gamma, weights, dtype=np.float32):
    A = np.zeros((2*d,2*d), dtype=dtype)
    B = np.zeros((2*d,2*d), dtype=dtype)

    for i in range(d-1):
        qubit_index = d + j + (2 * d - 1) * i 
        w = weights[qubit_index] 
        log_gamma = log_gamma + np.log(1 + w**2)
        t = (2 * w)    / (1 + w**2) 
        s = (1 - w**2) / (1 + w**2)
        A[2*i+1, 2*i+2] = t
        A[2*i+2, 2*i+1] =-t
        B[2*i+1, 2*i+1] = s
        B[2*i+2, 2*i+2] = s
    log_gamma  = log_gamma +  np.log(np.sqrt(np.linalg.det(m + A)))
    # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
    # Step 1: Solve (M + A) * X = B for X
    x = np.linalg.solve(m + A, B)
    # Step 2: Compute B * X
    m = A - B @ x
    # m=a - (b @ np.linalg.inv(m + a) @ b)
    return m, log_gamma

@numba.njit
def coset_probability_log(d,p,f, dtype=np.float32):
    weights = calc_weights(p,f)

    m = gen_m0(d)
    log_gamma = np.log(2**(d-1))


    for j in range(d - 1):
        m, log_gamma = simulate_horizontal_log(d, j, m, log_gamma, weights, dtype=dtype)
        m, log_gamma = simulate_vertical_log(d, j, m, log_gamma, weights, dtype=dtype)
    m, log_gamma = simulate_horizontal_log(d, d-1, m, log_gamma, weights, dtype=dtype) # d-1 due to 0 <= i < d and not 1<=i<=d

    # repr error prob
    n = d**2 + (d - 1)**2
    norm_f = np.sum(f)
    pauli_error_prob = (1 - p)**(n - norm_f) * p**norm_f

    log_coset_prob = 1/2 * log_gamma - 1/2 * np.log(2) + np.log(pauli_error_prob)  + 1/4*np.log(np.linalg.det(m + gen_m_0(d)))
    return log_coset_prob

@numba.njit
def decode_half_syndrome_log(d, p, h_syndrome, stab_type="Z", dtype=np.float32):
    if stab_type.upper() == "Z":
        stabilizer_matrix = format_syndrome_to_matrix(d, h_syndrome)
    elif stab_type.upper() == "X":
        rot_synd = rotate_and_reorder_syndrome(d, h_syndrome)
        stabilizer_matrix = format_syndrome_to_matrix(d, rot_synd)
    else:
        raise ValueError("unexpected detector")
    
    # prob of coset without logical error 
    f, c_f = stabilizer_to_pauli(d, stabilizer_matrix)
    log_p_I = coset_probability_log(d, p, f, dtype=dtype)
    # prob of coset with logical error 
    f, _ = stabilizer_to_pauli(d, stabilizer_matrix, add_logical=True)
    log_p_L = coset_probability_log(d, p, f, dtype=dtype)

    obs_flip = True if log_p_I < log_p_L else False
    return obs_flip, c_f

# arons code adapted 
@numba.njit
def decode_half_syndrome_aron(d, p, h_syndrome, stab_type="Z",):
    if stab_type.upper() == "X":
        # need to rotate X stabilizer -> can be treated like Z stabilizer 
        h_syndrome = rotate_and_reorder_syndrome(d, h_syndrome)
    stabilizer_matrix = format_syndrome_to_matrix(d, h_syndrome)

    f_I, c_f = stabilizer_to_pauli(d, stabilizer_matrix)
    f_L, _ = stabilizer_to_pauli(d, stabilizer_matrix, add_logical=True)
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
    return obs_flip, c_f