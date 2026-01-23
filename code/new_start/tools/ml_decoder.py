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

# # Identity Matrix NOT NEEDED
# def gen_identity(d):
#     v = 2*d * [1]
#     i = sc.sparse.diags_array(v).toarray()
#     return i 


class ML_Decoder():
    def __init__(self, d, error_rate):
        self.distance = d
        self.d = d
        self.er = error_rate # TODO: adapt for X and Z errors
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
        detector_list = np.multiply(detector_list,1) # Boolean to int
        syndrome_array = np.reshape(detector_list, (self.d, self.d-1))
        return syndrome_array


    # method to generate a fitting pauli string (fitting to the detector)
    def syndrome_to_some_pauli(self, syndrome_array, add_logical: bool = False):
        # generate the pauli string from detector outputs
        # start with Horizontal ones (notation from eff ML paper)

        # not the same indexing as python !
        f = np.zeros((self.d,self.d)) # at this step only horizontal errors are taken into account
        #TODO optimize this step:
        for i, row in enumerate(syndrome_array):
            for j, detector in enumerate(row):
                if detector:
                    for jt in range(j+1):
                        f[i,jt] = (f[i,jt] + 1)%2
        if add_logical:
            # add logical operator by adding stabilizers to the first row 
            f[0,:] = (f[0,:] + 1) % 2  

        # add the vertical qubits erros to pauli string in 0 state (no error)
        f = np.concatenate((f,np.zeros((self.d,self.d-1))),axis=1)
        # flatten so that we can refer to them by index of qubit/edge location!
        f = f.flatten()
        f = f[:-(self.d-1)] # last row of verticals does not exists!
        self.f = f 
        self.__calc_weights()
        return f
    
    def __calc_weights(self,):
        # TODO I am assuming global equal error rates (not local to qubits!)
        self.weights = self.er**(1-2*self.f) * (1-self.er)**(-1+2*self.f) 

    ## Algorithm 1 
    def __simulate_horizontal(self,j,m,gamma):
        qubit_indices = j + (2 * self.d - 1) * np.arange(0,self.d)
        # subset of weights relevant 
        ws = self.weights[qubit_indices]
        for w in ws:
            # iterative
            gamma *= (1 + w**2) / 2
        # safe some time by using numpy
        ts = (1 - ws**2) / (1 + ws**2) 
        ss = (2 * ws) / (1 + ws**2) 
        # Anti-symmetric matrix
        v1 = np.zeros(2 * self.d - 1)
        v1[::2] = ts
        a = gen_anti_symmetric_matrix(v1)
        # Diagonal matrix
        v2 = np.repeat(ss,2) 
        b = gen_diagnoal_matrix(v2)
        # Final calc:
        gamma  *= np.sqrt(np.linalg.det(m + a))
        # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
        # Step 1: Solve (M + A) * X = B for X
        x = np.linalg.solve(m + a, b)
        # Step 2: Compute B * X
        m = a - b @ x
        return m, gamma


    def __simulate_vertical(self,j,m,gamma):
        qubit_indices = self.d + j + (2 * self.d - 1) * np.arange(self.d - 1)  
        # subset of weights relevant 
        ws = self.weights[qubit_indices]
        for w in ws:
            # iterative
            gamma *= (1 + w**2)
        # safe some time by using numpy
        ts = (2 * ws) / (1 + ws**2) 
        ss = (1 - ws**2) / (1 + ws**2) 
        # Anti-symmetric matrix
        v1 = np.zeros(2 * self.d - 1) # TODO NOT sure if correct
        v1[1::2] = ts
        a = gen_anti_symmetric_matrix(v1)
        # Diagonal matrix
        v2 = np.append(np.append([1],np.repeat(ss,2)),[1]) # not clean but = [1,s0,s0,...,sd-1,sd-1,1] 
        b = gen_diagnoal_matrix(v2)
        # Final calc:
        gamma  *= np.sqrt(np.linalg.det(m + a))
        # Avoid using direct calculations of inverse, as it might be more unstable ... https://nhigham.com/2022/03/28/what-is-the-matrix-inverse/?utm_source=chatgpt.com
        # Step 1: Solve (M + A) * X = B for X
        x = np.linalg.solve(m + a, b)
        # Step 2: Compute B * X
        m = a - b @ x
        return m, gamma
    
    def coset_probability(self,):
        # it uses the weight indirectly over the object

        m = gen_m_0(self.d)
        gamma = 2**(self.d - 1)

        for j in range(self.d - 1):
            m, gamma = self.__simulate_horizontal(j,m,gamma)
            m, gamma = self.__simulate_vertical(j,m,gamma)
        m, gamma = self.__simulate_horizontal(self.d - 1,m,gamma) # d-1 due to 0 <= i < d and not 1<=i<=d

        # calc error probability of the arb choosen error 
        n = self.d**2 + (self.d - 1)**2
        norm_f = np.sum(self.f)
        pauli_error_prob = (1 - self.er)**(n - norm_f) * self.er**norm_f

        coset_prob = pauli_error_prob * np.sqrt(gamma / 2) * (np.linalg.det((m + gen_m_0(self.d))))**(1/4)
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
        prob_coset_1 = self.coset_probability() 

        # prob of coset with logical error 
        self.syndrome_to_some_pauli(syndrome_array,add_logical=True)
        prob_coset_error = self.coset_probability() 

        if prob_coset_1 > prob_coset_error:
            return False # no observable flip predicted 
        elif prob_coset_1 < prob_coset_error:
            return True # observable flip predicted
        else: 
            print("unexpected")
            return 0
        



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
        False, False,
        False, False,
        False, False,
    ]
    detector_pauli = "X"
    distance = 3
    syndrome_list = syndrome_list3
    # error rate -> 0 => coset probability -> 1
    # TODO check if this is reasonable 
    error_rate = 0.5 
    decoder = ML_Decoder(distance, error_rate)
    print(decoder.decode_syndrome(syndrome_list,detector_pauli))
