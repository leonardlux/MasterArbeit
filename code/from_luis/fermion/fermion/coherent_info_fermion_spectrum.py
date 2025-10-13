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
    if not np.allclose(A, -A.T):
        raise ValueError("Matrix must be antisymmetric.")
    
    # Ensure matrix dimensions are even
    n = A.shape[0]
    if n % 2 != 0:
        raise ValueError("Matrix must have an even dimension to compute the Pfaffian.")
    
    # Compute the Pfaffian using pfapack, which returns the correct sign
    pf = pfaffian.pfaffian(A)
    
    return pf
    

def pfaffian_recursive(A):
    """
    Computes the Pfaffian of an antisymmetric matrix A using a recursive approach.
    
    Parameters:
    - A: np.ndarray, an antisymmetric matrix of even dimension (n x n).
    
    Returns:
    - float: Pfaffian of the matrix.
    """
    # Ensure A is square and has even dimensions
    n = A.shape[0]
    if n % 2 != 0:
        raise ValueError("Matrix must be of even dimensions to compute the Pfaffian.")
    if n == 0:
        return 1
    elif n == 2:
        return A[0, 1]
    
    # Recursive computation
    pf = 0
    for j in range(1, n):
        # Create a minor matrix by excluding rows and columns 0 and j
        minor = np.block([
            [A[1:j, 1:j], A[1:j, j+1:n]],
            [A[j+1:n, 1:j], A[j+1:n, j+1:n]]
        ])
        
        # Update the Pfaffian sum
        pf += (-1)**(j+1) * A[0, j] * pfaffian_recursive(minor)
    
    return pf
    



def build_hamiltonian(Lx,Ly,t,bcx,bcy):

    #Lx spin-length on x direction
    #Ly spin-length on y direction
    #bcx 0: periodic 1: anti-periodic on X
    #bcy 0: periodic 1: anti-periodic on Y
    
    N = 4*Lx*Ly
    lx = 2
    ly = 2
    
    H = np.zeros((N,N))
    
    H = np.reshape(H,(Lx*Ly,lx*ly,Lx*Ly,lx*ly))
    
    #up,right,down,left
    #Hint = np.array([[0,-1,+1,-1],[+1,0,+1,-1],[-1,-1,0,+1],[1,+1,-1,0]])
    #Hint = Hint.T
    #print(Hint)
    #Hint = np.array([[0,+1,-1,-1],[-1,0,-1,+1],[1,1,0,-1],[-1,-1,+1,0]])
    
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
    
    #Hint = np.reshape(Hint,(2,2,2,2))

    for x in range(Lx):
    
        for y in range(Ly):
        
            idx = lattice_to_integer(x, y,Ly)
        
            #H[idx,0,0,idx,0,0] = 0.0
            
            H[idx,:,idx,:] += Hint
            
            #Neighbours
            
            idx2 = lattice_to_integer((x+1)%Lx,y,Ly)
            res = bcx * (x+1)//Lx
            #H[idx,0,1,idx2,1,1] += t * (-1)**(res)
            H[idx,1,idx2,3] -= t * (-1)**(res)
            
            idx2 = lattice_to_integer((x-1)%Lx,y,Ly)
            res = bcx * (x-1)//Lx
            #H[idx,1,1,idx2,0,1] -= t * (-1)**(res)
            H[idx,3,idx2,1] += t * (-1)**(res)
            
            idx2 = lattice_to_integer(x,(y+1)%Ly,Ly)
            res = bcy * (y+1)//Ly
            #H[idx,0,0,idx2,1,0] -= t * (-1)**(res)
            H[idx,0,idx2,2] += t * (-1)**(res)
            
            idx2 = lattice_to_integer(x,(y-1)%Ly,Ly)
            res = bcy * (y-1)//Ly
            #H[idx,1,0,idx2,0,0] += t  * (-1)**(res)
            H[idx,2,idx2,0] -= t  * (-1)**(res)
        
         
    return 0.5*np.reshape(H,(N,N))
   
def topo_invariant(Hpp,Hpa,Hap,Haa):

    pp = compute_pfaffian(Hpp)
    ap = compute_pfaffian(Hap)
    pa = compute_pfaffian(Hpa)
    aa = compute_pfaffian(Haa)
    
    #print(pp,ap,pa,aa)
    #print(pp+ap+pa+aa)
    
    ti = 1-2*pp/(pp+ap+pa+aa)
    return ti

p=0.2
Lx=1
Ly=1
#t=1-2*p
t = np.square(1-2*p)
print("t = ", t)

Hpp = build_hamiltonian(Lx,Ly,t,0,0)
Hpa = build_hamiltonian(Lx,Ly,t,0,1)
Hap = build_hamiltonian(Lx,Ly,t,1,0)
Haa = build_hamiltonian(Lx,Ly,t,1,1)

#print(Hpp)
#print(Hpa)
#print(Hap)
#print(Haa)
#print(np.linalg.norm(Hpp+Hpp.T) )

t1 = time.time()
print("pfaffian = ", compute_pfaffian(Haa))
t2 = time.time()
print("Time determinant = ", (t2-t1)/60, "minutes")

ti_val = topo_invariant(Hpp,Hpa,Hap,Haa)
print("topo invariant ",ti_val)
print("coherent information = ", 2*np.log2(ti_val))




#PP = 0.5*np.array([[0,2,-1,-1],[-2,0,-1,1],[1,1,0,-2],[1,-1,2,0]])
#AA = 0.5*np.array([[0,0,-1,-1],[0,0,-1,1],[1,1,0,0],[1,-1,0,0]])

#AP = 0.5*np.array([[0,2,-1,-1],[-2,0,-1,1],[1,1,0,0],[1,-1,0,0]])
#PA = 0.5*np.array([[0,0,-1,-1],[0,0,-1,1],[1,1,0,-2],[1,-1,2,0]])

#ti_val = topo_invariant(PP,PA,AP,AA)



#print("Coherent information single qubit")
#print(2*np.log2(ti_val))


ps = np.linspace(0,0.5,80)
#ps = np.linspace(np.sqrt(2)-1-0.02,np.sqrt(2)-1+0.02,10)

ts = np.zeros(ps.shape[0])

#co_info = np.zeros(ps.shape[0])

Ls = np.array([1],dtype="int")

energies = np.zeros((ps.shape[0],4*Ls[0]**2))

for l,ll in enumerate(Ls):
    ll=int(ll)
    for idx,p in enumerate(ps):

        t=np.square(1-2*p)
        #t=p
        ts[idx] = t
    
        Hpp = build_hamiltonian(ll,ll,t,0,0)
        #Hap = build_hamiltonian(ll,ll,t,0,1)
        #Hpa = build_hamiltonian(ll,ll,t,1,0)
        #Haa = build_hamiltonian(ll,ll,t,1,1)
        e = np.linalg.norm(Hpp+Hpp.T)
    
        co_info[idx,:] = e
   
    
Is = 4*np.log2(np.square(1-ps)+np.square(ps))+2


pyplot.plot(ps,co_info[:,l],label="d={}".format(ll))
    
pyplot.plot(ps,Is,"--")

#pyplot.axvline(x=np.sqrt(2)-1, color="grey",label="$\sqrt{2}-1$")
pyplot.legend()

pyplot.show()

