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
    

def kitaev_ham(Nsites,t,Delta,mu,l):

    Nprime = 2*Nsites
    Hmat = np.zeros([Nprime,Nprime])        # Declare a 2Nx2N matrix
    #Jx = 0.5*(params['t'] - params['Delta'])
    #Jy = 0.5*(params['t'] + params['Delta'])
    Jx = 0.5*(t - Delta)
    Jy = 0.5*(t + Delta)
    
    for n in range(Nsites-1):
        Hmat[2*n,2*n+1]   = Jx
        Hmat[2*n+1,2*n]   = -Jx
        
        Hmat[2*n-1,2*n+2] = -Jy
        Hmat[2*n+2,2*n-1] = Jy
        
        Hmat[2*n-1,2*n]   = mu
        Hmat[2*n,2*n-1]   = -mu

    Hmat[2*(Nsites-1)-1,2*(Nsites-1)] = mu
    Hmat[2*(Nsites-1),2*(Nsites-1)-1] = -mu
    
    #Hmat[0,2*Nsites-1] = l*Jx
    #Hmat[2*Nsites-1,0] = -l*Jx
    
    Hmat[2*Nsites-3,0] = -l*Jy
    Hmat[0,2*Nsites-3] = l*Jy
    
    Hmat = 1j*Hmat

    return Hmat



   
def topo_invariant(Hp,Ha):

    p = compute_pfaffian(Hp)
    a = compute_pfaffian(Ha)
   
    
    #ti = np.sign(1-p/(a+p))
    ti = np.sign(p*a)
    return ti

#Nsites = 25              # Number of lattice sites
#Nprime = 2*Nsites
#e_threshold = 1E-6     # Threshold for finding zero eigenstates
#params = {
#'t' : 2.0,               # Nearest neighbor hopping
#'Delta' : 2.0,           # Superconducting pairing term
#'mu' : 0.0               # Chemical potential
#}

t=2.0
Delta=2.0



#Hpp = build_hamiltonian(Lx,Ly,t,0,0)
#Hpa = build_hamiltonian(Lx,Ly,t,0,1)
#Hap = build_hamiltonian(Lx,Ly,t,1,0)
#Haa = build_hamiltonian(Lx,Ly,t,1,1)

#print(Hpp)
#print(Hpa)
#print(Hap)
#print(Haa)
#print(np.linalg.norm(Hpp+Hpp.T) )

#t1 = time.time()
#print("pfaffian = ", compute_pfaffian(Haa))
#t2 = time.time()
#print("Time determinant = ", (t2-t1)/60, "minutes")

#ti_val = topo_invariant(Hpp,Hpa,Hap,Haa)
#print("topo invariant ",ti_val)
#print("coherent information = ", 2*np.log2(ti_val))




#PP = 0.5*np.array([[0,2,-1,-1],[-2,0,-1,1],[1,1,0,-2],[1,-1,2,0]])
#AA = 0.5*np.array([[0,0,-1,-1],[0,0,-1,1],[1,1,0,0],[1,-1,0,0]])

#AP = 0.5*np.array([[0,2,-1,-1],[-2,0,-1,1],[1,1,0,0],[1,-1,0,0]])
#PA = 0.5*np.array([[0,0,-1,-1],[0,0,-1,1],[1,1,0,-2],[1,-1,2,0]])

#ti_val = topo_invariant(PP,PA,AP,AA)



#print("Coherent information single qubit")
#print(2*np.log2(ti_val))


mus = np.linspace(0,4,30)
#ps = np.linspace(np.sqrt(2)-1-0.02,np.sqrt(2)-1+0.02,10)



#co_info = np.zeros(ps.shape[0])

Ls = np.array([6,10,20,30,40],dtype="int")

co_info = np.zeros((mus.shape[0],Ls.shape[0]))

for l,ll in enumerate(Ls):
    ll=int(ll)
    for idx,mu in enumerate(mus):

        #t=np.square(1-2*p)
        #t=p
        #ts[idx] = t
        
        Hp = kitaev_ham(ll,t,Delta,mu,1.0)
        Ha = kitaev_ham(ll,t,Delta,mu,-1.0)
        #Hpa = build_hamiltonian(ll,ll,t,1,0)
        #Haa = build_hamiltonian(ll,ll,t,1,1)
    
        co_info[idx,l] = topo_invariant(Hp,Ha)
   
    
#Is = 4*np.log2(np.square(1-ps)+np.square(ps))+2

for l,ll in enumerate(Ls):
    pyplot.plot(mus,co_info[:,l],"-*",label="N={}".format(ll))
    
#pyplot.plot(ps,Is,"--")
pyplot.ylabel("topological invariant")
pyplot.xlabel("mu")
#pyplot.axvline(x=np.sqrt(2)-1, color="grey",label="$\sqrt{2}-1$")
pyplot.legend()

pyplot.show()

