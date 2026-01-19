import numpy as np
from pfapack import pfaffian
import time
import matplotlib.pyplot as pyplot
from numba import njit, prange


def lattice_to_integer(row, col, n_cols):
    """
    Converts coordinates (row, col) in a square lattice to a unique integer.
    
    Parameters:
    - row: int, row index in the lattice.
    - col: int, column index in the lattice.
    - n_cols: int, total number of columns in the lattice.
    
    Returns:
    - int: unique integer representing the position (row, col) in the lattice.
    """
    return row * n_cols + col
    

def compute_pfaffian(A):
    """
    Computes the Pfaffian of an antisymmetric matrix A with the correct sign.
    
    Parameters:
    - A: np.ndarray, an antisymmetric matrix of even dimension (n x n).
    
    Returns:
    - float: Pfaffian of the matrix with the correct sign.
    """
    # Check if matrix is antisymmetric
    #if not np.allclose(A, -A.T):
    #    raise ValueError("Matrix must be antisymmetric.")
    
    # Ensure matrix dimensions are even
    #n = A.shape[0]
    #if n % 2 != 0:
    #    raise ValueError("Matrix must have an even dimension to compute the Pfaffian.")
    
    # Compute the Pfaffian using pfapack, which returns the correct sign
    pf = pfaffian.pfaffian(A)
    
    return pf

@njit
def build_hamiltonian(Lx,Ly,t,bcx,bcy,disorder_conf):

    #Lx spin-length on x direction
    #Ly spin-length on y direction
    #bcx 0: periodic 1: anti-periodic on X
    #bcy 0: periodic 1: anti-periodic on Y
    #disorder_conf : array of size 2LxLy with entires. The first LxLy are horizontal and the second LxLy are vertical
    
    N = 4*Lx*Ly
    lx = 2
    ly = 2
    
    H = np.zeros((N,N))
    
    H = np.reshape(H,(Lx*Ly,lx*ly,Lx*Ly,lx*ly))
    
    Hint = np.zeros((4,4))
    
    Hint[0,2] = +1
    Hint[2,0] = -1
    
    Hint[0,1] =-1
    Hint[1,0] =+1
    
    Hint[0,3] =-1
    Hint[3,0] =+1
    
    Hint[1,2] =+1
    Hint[2,1] =-1
    
    Hint[1,3] =-1
    Hint[3,1] =+1
    
    Hint[2,3] =+1
    Hint[3,2] =-1
    
    eta_horiz = 1-2*disorder_conf[0:Lx*Ly]
    eta_vertical = 1-2*disorder_conf[Lx*Ly:2*Lx*Ly]
    
    #print(eta_horiz)
    #print(eta_vertical)
    
    for idx in prange(Lx * Ly):
    
        x = idx // Ly  # Row index (first loop index)
        y = idx % Ly

  
        #idx = lattice_to_integer(x, y,Ly)
        #idx = x * Ly + y
            
        H[idx,:,idx,:] += Hint
            
            
        #Neighbours
            
        #idx2 = lattice_to_integer((x+1)%Lx,y,Ly)
        idx2 = (x+1)%Lx * Ly + y
        res = bcx * (x+1)//Lx
        #idx_eta = lattice_to_integer((x+1)%Lx,y,Ly)
        #print(idx,idx2)
        eta = eta_horiz[idx2]
        #print(eta)
        H[idx,1,idx2,3] -= t * (-1)**(res) * eta
            
        #idx2 = lattice_to_integer((x-1)%Lx,y,Ly)
        idx2 = (x-1)%Lx * Ly + y
        res = bcx * (x-1)//Lx
        #idx_eta = lattice_to_integer(x,y,Ly)
        #print(idx,idx2)
        eta = eta_horiz[idx]
        #print(eta)
        H[idx,3,idx2,1] += t * (-1)**(res) * eta
            
        #idx2 = lattice_to_integer(x,(y+1)%Ly,Ly)
        idx2 = x * Ly + (y+1)%Ly
        res = bcy * (y+1)//Ly
        #idx_eta = lattice_to_integer(x,y,Ly)
        eta = eta_vertical[idx]
        H[idx,0,idx2,2] += t * (-1)**(res) * eta
            
        #idx2 = lattice_to_integer(x,(y-1)%Ly,Ly)
        idx2 = x * Ly + (y-1)%Ly
        res = bcy * (y-1)//Ly
        #idx_eta = lattice_to_integer(x,(y-1)%Ly,Ly)
        eta = eta_vertical[idx2]
        H[idx,2,idx2,0] -= t  * (-1)**(res) * eta
        
         
    return 0.5*np.reshape(H,(N,N))
 

def topo_invariant(Hpp,Hpa,Hap,Haa):

    pp = compute_pfaffian(Hpp)
    ap = compute_pfaffian(Hap)
    pa = compute_pfaffian(Hpa)
    aa = compute_pfaffian(Haa)
    
    #print(pp,ap,pa,aa)
    
    ti = 1-2*pp/(pp+ap+pa+aa)
    return ti
    
def topo_invariant_2(Hpp,Hpa,Hap,Haa):

    pp = compute_pfaffian(Hpp)
    #ap = compute_pfaffian(Hap)
    #pa = compute_pfaffian(Hpa)
    #aa = compute_pfaffian(Haa)
    
    #print(pp,ap,pa,aa)
    
    ti = np.sign(pp)
    return -2*ti
    

def average_topo_invariant(Lx,Ly,p):


    n_qubits = 2*Lx*Ly
    t = 1-2*p
    
    P1 = 1-p
    P2 = p
    
    ci = 0.0
    
    for x in range(2**n_qubits):
    
        
        #array_errors = int_to_binary_array(x, bit_length=n_qubits)
        array_errors = np.array([int(i) for i in bin(x)[2:].zfill(n_qubits)])
        
        
        #print(array_errors)
        
        n_minus = np.sum(array_errors)
        n_plus = n_qubits-n_minus
        
        #t1 = time.time()
        Hpp = build_hamiltonian(Lx,Ly,t,0,0,array_errors)
        Hpa = build_hamiltonian(Lx,Ly,t,0,1,array_errors)
        Hap = build_hamiltonian(Lx,Ly,t,1,0,array_errors)
        Haa = build_hamiltonian(Lx,Ly,t,1,1,array_errors)
        #t2 = time.time()
        #if x==0: print("Building hamiltonian took = ", (t2-t1)/60. )
        #print(Hpp)
        #print(Hpp)
        #t1 = time.time()
        ti_val = topo_invariant(Hpp,Hpa,Hap,Haa)
        #ti_val = topo_invariant_2(Hpp,Hpa,Hap,Haa)
        #t2 = time.time()
        #if x==0: print("Topo invariant took = ", (t2-t1)/60. )
        
        Prob = np.exp(n_plus*np.log(P1)+n_minus*np.log(P2))
        
        ci += Prob * 2 * np.log2(ti_val)
        #ci += Prob * ti_val
        
        
        
        #print(np.log2(ti_val))


    return ci

#p=0.001
#Lx=1
#Ly=1

#ci = average_topo_invariant(Lx,Ly,p)

#print("Coherent information = ",ci)




ps = np.linspace(1e-4,0.5,10)
#ps = np.array([0.1])
#ps = np.linspace(np.sqrt(2)-1-0.02,np.sqrt(2)-1+0.02,10)

ts = np.zeros(ps.shape[0])

#co_info = np.zeros(ps.shape[0])

Ls = np.array([1,2],dtype="int")

co_info = np.zeros((ps.shape[0],Ls.shape[0]))

for l,ll in enumerate(Ls):
    ll=int(ll)
    for idx,p in enumerate(ps):
    
        #t1 = time.time()
        ci = average_topo_invariant(ll,ll+1,p)
        #t2 = time.time()
        #print("Computation took = ", (t2-t1)/60., " seconds")
        co_info[idx,l] = ci
   
    
#Is = 4*np.log2(np.square(1-ps)+np.square(ps))+2
Is = 4*( ps*np.log2(ps)+(1-ps)*np.log2(1-ps))+2

#print(Is)
print(co_info[:,0])

for l,ll in enumerate(Ls):
    pyplot.plot(ps,co_info[:,l],label="d={}".format(ll))
    
    
pyplot.plot(ps,Is,"--")

#pyplot.axvline(x=np.sqrt(2)-1, color="grey",label="$\sqrt{2}-1$")
pyplot.legend()

pyplot.show()


