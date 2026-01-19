import numpy as np
from pfapack import pfaffian
import time




def pfaffian_via_determinant(A):
    # Check if A is square and of even dimension
    n = A.shape[0]
    if n % 2 != 0:
        raise ValueError("Matrix must be even-dimensional to compute the Pfaffian.")
    
    # Calculate the determinant and take the square root
    det_A = np.linalg.det(A)
    pf_A = np.sqrt(det_A)
    
    # Adjust sign for numerical stability
    return pf_A if det_A >= 0 else -pf_A


def topo_invariant(Hpp,Hpa,Hap,Haa):

    pp = pfaffian_via_determinant(Hpp)
    ap = pfaffian_via_determinant(Hap)
    pa = pfaffian_via_determinant(Hpa)
    aa = pfaffian_via_determinant(Haa)
    
    print(pp,ap,pa,aa)
    print(pp+ap+pa+aa)
    
    ti = 1-2*pp/(pp+ap+pa+aa)
    
    return ti

#p=0.0
#Lx=1
#Ly=1
#t=1-2*p
#t = np.square(1-2*p)
#print("t = ", t)

PP = 0.5*np.array([[0,2,-1,-1],[-2,0,-1,1],[1,1,0,-2],[1,-1,2,0]])
AA = 0.5*np.array([[0,0,-1,-1],[0,0,-1,1],[1,1,0,0],[1,-1,0,0]])

AP = 0.5*np.array([[0,2,-1,-1],[-2,0,-1,1],[1,1,0,0],[1,-1,0,0]])
PA = 0.5*np.array([[0,0,-1,-1],[0,0,-1,1],[1,1,0,-2],[1,-1,2,0]])

ti_val = topo_invariant(PP,PA,AP,AA)



print("Coherent information single qubit")
print(2*np.log2(ti_val))
