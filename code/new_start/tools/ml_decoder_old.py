from aron_ml import error_converter_c, simulation_mld 
import numpy as np
import scipy as sc

## Matrix definition

# Matrix A
def gen_anti_symmetric_matrix(l):
    """
    return antisymetric matrix with l and -l on the diagonals

    :param l: numpy array representing a vector which is a element of R^m 
    """
    rows = [np.array(l), -1*np.array(l)]
    diags = [1,-1]
    a = sc.sparse.diags_array(rows, offsets = diags).toarray()
    return a

# Matrix D
def gen_diagnoal_matrix(l):
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

    # method to generate a fitting pauli string (fitting to the detector output)
def syndrome_to_some_pauli(d, syndrome_array, add_logical: bool = False):
    # generate the pauli string from detector outputs
    # start with Horizontal ones (notation from eff ML paper)

    # not the same indexing as python !
    f = np.zeros((d,d)) # at this step only errors on horizontal qubits are generated 
    #TODO optimize this step:
    for i, row in enumerate(syndrome_array):
        for j, detector in enumerate(row):
            if detector:
                for jt in range(j+1):
                    f[i,jt] = (f[i,jt] + 1)%2
    if add_logical:
        # add logical operator by inverting first row (applying a logical gate) 
        f[0,:] = (f[0,:] + 1) % 2  

    # add the vertical qubits erros to pauli string in 0 state (no error)
    f = np.concatenate((f,np.zeros((d,d-1))),axis=1)
    # flatten so that we can refer to them by index of qubit/edge location!
    f = f.flatten()
    f = f[:-(d-1)] # last row of verticals does not exists!
    return f

class ML_Decoder():
    def __init__(self, d, error_rate):
        self.d = d
        self.p = error_rate # TODO: adapt for X and Z errors
        pass

    # How to I integrate those cleanly?
    def rotate_X_stab(self, detector_list):    
        detector_list = np.multiply(detector_list,1) # Boolean to int
        syndrome_array = np.zeros((self.d, self.d-1)) # syndrome array is the shaped and correctly ordered detector
        for col in range(self.d-1):
            for row in range(self.d):
                syndrome_array[row,-1* (col + 1)] = detector_list[ col * self.d + row ]
        return syndrome_array 
    
    def format_Z_stab(self, detector_list):
        """
        from list to matrix shaped like the qubits location (d,d-1) 
        """
        detector_list = np.multiply(detector_list,1) # Boolean to int
        syndrome_array = np.reshape(detector_list, (self.d, self.d-1))
        return syndrome_array

    def calc_weights(self,):
        self.weights = self.p**(1-2*self.f) * (1-self.p)**(-1+2*self.f) 



    ## Algorithm 1 
    def simulate_horizontal(self,j,m,gamma,log_gamma):
        qubit_indices = j + (2 * self.d - 1) * np.arange(0,self.d)
        # subset of weights relevant 
        ws = self.weights[qubit_indices]
        # iterative
        for w in ws:
            gamma = gamma * (1 + w**2) / 2 
        log_gamma += np.sum(np.log((1+ws**2)/2))
        # broadcasting
        t = (1 - ws**2) / (1 + ws**2) 
        s = (2 * ws) / (1 + ws**2) 
        # generation of A 
        v1 = np.zeros(2 * self.d - 1)
        v1[::2] = t
        a = gen_anti_symmetric_matrix(v1)
        # generation of B 
        v2 = np.repeat(s,2) 
        b = gen_diagnoal_matrix(v2)
        # Final calc:
        gamma  = gamma * np.sqrt(np.linalg.det(m + a))
        log_gamma += np.log(np.sqrt(np.linalg.det(m+a)))
        # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
        # Step 1: Solve (M + A) * X = B for X
        # x = np.linalg.solve(m + a, b)
        # Step 2: Compute B * X
        # m = a - b @ x @ b
        m = a - (b @ np.linalg.inv(m + a) @ b)
        return m, gamma, log_gamma

    def simulate_vertical(self,j,m,gamma, log_gamma):
        qubit_indices = self.d + j + (2 * self.d - 1) * np.arange(self.d - 1)  
        # subset of weights relevant 
        ws = self.weights[qubit_indices]
        for w in ws:
            # iterative (TODO does not need to be iterative!)
            gamma = gamma * (1 + w**2)
        log_gamma += np.sum(np.log(1+ws**2))
        # safe some time by using numpy
        ts = (2 * ws) / (1 + ws**2) 
        ss = (1 - ws**2) / (1 + ws**2) 
        # Anti-symmetric matrix
        v1 = np.zeros(2 * self.d - 1) 
        v1[1::2] = ts
        a = gen_anti_symmetric_matrix(v1)
        # Diagonal matrix
        v2 = np.append(np.append([1],np.repeat(ss,2)),[1]) # not clean but = [1,s0,s0,...,sd-1,sd-1,1] 
        b = gen_diagnoal_matrix(v2)
        # Final calc:
        gamma  = gamma * np.sqrt(np.linalg.det(m + a))
        log_gamma += np.log(np.sqrt(np.linalg.det(m + a)))
        # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
        # Step 1: Solve (M + A) * X = B for X
        # x = np.linalg.solve(m + a, b)
        # Step 2: Compute B * X
        # m = a - b @ x
        m=a - (b @ np.linalg.inv(m + a) @ b)
        return m, gamma, log_gamma
    
    def coset_probability(self,syndrome):
        syndrome_to_some_pauli(self.d,syndrome)

        m = gen_m_0(self.d)
        gamma = 2**(self.d - 1)
        log_gamma = np.log(2**(self.d-1))

        # repr error prob
        n = self.d**2 + (self.d - 1)**2
        norm_f = np.sum(self.f)
        pauli_error_prob = (1 - self.p)**(n - norm_f) * self.p**norm_f

        for j in range(self.d - 1):
            m, gamma, log_gamma = self.simulate_horizontal(j,m,gamma, log_gamma)
            m, gamma, log_gamma = self.simulate_vertical(j,m,gamma, log_gamma)
        m, gamma, log_gamma = self.simulate_horizontal(self.d-1,m,gamma, log_gamma) # d-1 due to 0 <= i < d and not 1<=i<=d
        # calc error probability of the arb choosen error 
        coset_prob = pauli_error_prob * np.sqrt(gamma / 2) * (np.linalg.det((m + gen_m_0(self.d))))**(1/4)
        log_coset_prob = log_gamma -2  + np.log(pauli_error_prob * (np.linalg.det(m + gen_m_0(self.d)))**(1/4))
        # log_coset_prob does not make a difference
        return coset_prob 
    
    def decode_syndrome(self, detector_list, detector_pauli):
        if detector_pauli.upper() == "X":
            syndrome_array = self.rotate_X_stab(detector_list)
        elif detector_pauli.upper() == "Z":
            syndrome_array = self.format_Z_stab(detector_list)
        else:
            raise ValueError("unexpected detector")

        # prob of coset without logical error 
        self.syndrome_to_some_pauli(syndrome_array)
        p_I = self.coset_probability() 

        # prob of coset with logical error 
        self.syndrome_to_some_pauli(syndrome_array,add_logical=True)
        p_L = self.coset_probability() 

        if p_I < p_L:
            return True # observable flip predicted
        else: 
            return False # no oberservable flip predicted 


### Tests 
if __name__ == "__main__":
    syndrome_list5 = [
        True, False, False, False,
        False, False, False, False,
        False, False, False, False,
        False, False, False, False,
        False, False, False, True,
    ]
    syndrome_list3 = [
        True, False,
        False, True,
        True, True,
    ]
    detector_pauli = "Z"
    distance = 3
    syndrome_list = syndrome_list3 
    # error rate -> 0 => coset probability -> 1
    # TODO check if this is reasonable 
    error_rate = 0.1 
    decoder = ML_Decoder(distance, error_rate)
    print(decoder.decode_syndrome(syndrome_list,detector_pauli))
    print(test(d))
