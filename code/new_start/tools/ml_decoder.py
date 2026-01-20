import numpy as np
import scipy as sc

## Input Syndrome to some fixed Pauli
"""
input syndrome is a list of length (d*(d-1)) with boolean entries representing the dectetor values

I need to construct the physical qubits surface code (shape) to generate the error.

TODO: What is the difference between X and Y errors?!

"""
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

# Identity Matrix
def gen_identity(d):
    v = 2*d * [1]
    i = sc.sparse.diags_array(v).toarray()
    return i 


class ML_Decoder():
    def __init__(self, d, error_rate):
        self.distance = d
        self.d = d
        self.er = error_rate
        pass

    # method to generate a fitting pauli string(fitting to the detector)
    def __syndrome_to_some_pauli(self,detector_list):
        """
        Docstring for syndrome_to_pauli
        
        :param detector_list: list of detector events as an array of length (d*(d-1)) 
        :param distance: distance of the surface code 
        :return: list of pauli string, (not shaped) ordered by qubit/edge index 
        """
        if len(detector_list) != self.d * (self.d-1):
            raise ValueError("dector list has unexpected length")


        # TODO generalize to Z-Errors! start with X-errors
        # goal: generate the pauli error string

        # reshape the detector list to the correct form 
        reshape_X = (self.d-1,self.d) #TODO this way seems counter intuitive
        reshape_Z = (self.d,self.d-1) 
        shape = reshape_Z
        detector_ordered = np.reshape(detector_list, shape) 

        # generate the pauli string from detector outputs
        # start with Horizontal ones (notation from eff ML paper)

        # not the same indexing as python !
        f = np.zeros((self.d,self.d)) # at this step only horizontal errors are taken into account
        #TODO optimize this step:
        for i, row in enumerate(detector_ordered):
            for j, detector in enumerate(row):
                if detector:
                    for jt in range(j+1):
                        f[i,jt] = (f[i,jt]+1)%2

        # add the vertical qubits erros to pauli string in 0 state (no error)
        f = np.concatenate((f,np.zeros((self.d,self.d-1))),axis=1)
        # flatten so that we can refer to them by index of qubit/edge location!
        f = f.flatten()
        f = f[:-(self.d-1)] # last row of verticals does not exists!
        self.f = f 
        return f
    
    def calc_weights(self,):
        # TODO I am assuming global equal error rates (not local to qubits!)
        self.weights = self.er**(1-2*self.f) * (1-self.er)**(-1+2*self.f) 

    ## Algorithm 1 
    def simulate_horizontal(self,j,m,gamma):
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


    def simulate_vertical(self,j,m,gamma):
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
    
    def coset_probability(self,detector_list):
        self.__syndrome_to_some_pauli(detector_list)
        self.calc_weights()
        
        m = gen_m_0(self.d)
        gamma = 2**(self.d - 1)

        for j in range(self.d - 1):
            m, gamma = self.simulate_horizontal(j,m,gamma)
            m, gamma = self.simulate_vertical(j,m,gamma)
        m, gamma = self.simulate_horizontal(self.d - 1,m,gamma) # d-1 due to 0 <= i < d and not 1<=i<=d

        # calc error probability of the arb choosen error 
        n = self.d**2 + (self.d - 1)**2
        norm_f = np.sum(self.f)
        pauli_error_prob = (1 - self.er)**(n - norm_f) * self.er**norm_f

        coset_prob = pauli_error_prob * np.sqrt(gamma / 2) * (np.linalg.det((m + gen_m_0(self.d))))**(1/4)
        return coset_prob


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
    distance = 5
    syndrome_list = syndrome_list5
    error_rate = 0.1
    decoder = ML_Decoder(distance, error_rate)
    print(decoder.coset_probability(syndrome_list))

    # Seems to work ... now I need to implement it 

