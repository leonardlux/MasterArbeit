import numpy as np
import scipy as sc

## Input Syndrom to some fixed Pauli

# TODO



## Define the needed Matrices

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
    m_0 = sc.sparse.diags_array(rows, offsets = diags).toarray()
    m_0[0][-1] = 1
    m_0[-1][0] = -1
    return m_0

# Identity Matrix
def gen_identity(d):
    v = 2*d * [1]
    i = sc.sparse.diags_array(v).toarray()
    return i 

