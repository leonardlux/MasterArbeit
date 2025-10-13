import numpy as np
from pfapack import pfaffian
import time
#import matplotlib.pyplot as pyplot
#from numba import njit, prange
import random
from scipy import stats
#from pfapack.ctypes import pfaffian as cpf
from scipy.stats import binom
import sys
import h5py

def simple_bootstrap(x, f=np.mean, c=0.67, r=100):
    """ Use bootstrap resampling to estimate a statistic and
    its uncertainty.

    x (1d array): the data
    f (function): the statistic of the data we want to compute
    c (float): confidence interval in [0, 1]
    r (int): number of bootstrap resamplings

    Returns estimate of stat, lower error bar, upper error bar.
    """
    assert 0 <= c <= 1, 'Confidence interval must be in [0, 1].'
    # number of samples
    n = len(x)
    #print(x.shape)
    # stats of resampled datasets
    fs = np.asarray(
        [f(x[np.random.randint(0, n, size=n)]) for _ in range(r)]
    )
    # estimate and upper and lower limits
    med = 50 #median of the data
    val = np.percentile(fs, med)
    high = np.percentile(fs, med * (1 + c))
    low = np.percentile(fs, med * (1 - c))
    # estimate and uncertainties
    return high - low



 
    
#@njit(parallel=True)
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
    
    for idx in range(Lx * Ly):
    
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
 

def topo_invariant(Lx,Ly,p,array_errors):

    t = 1-2*p
    
    #t1 = time.time()
    Hpp = build_hamiltonian(Lx,Ly,t,0,0,array_errors)
    Hpa = build_hamiltonian(Lx,Ly,t,0,1,array_errors)
    Hap = build_hamiltonian(Lx,Ly,t,1,0,array_errors)
    Haa = build_hamiltonian(Lx,Ly,t,1,1,array_errors)
    #t2 = time.time()
    #if x==0: print("Building hamiltonian took = ", (t2-t1)/60. )
    
    pp = pfaffian.pfaffian(Hpp)
    ap = pfaffian.pfaffian(Hap)
    pa = pfaffian.pfaffian(Hpa)
    aa = pfaffian.pfaffian(Haa)
    
    #pp = pfaffian.pfaffian(Hpp,method="H")
    #ap = pfaffian.pfaffian(Hap,method="H")
    #pa = pfaffian.pfaffian(Hpa,method="H")
    #aa = pfaffian.pfaffian(Haa,method="H")
    
    #print(pp,ap,pa,aa)
        
    #t1 = time.time()
    ti_val = 1-2*pp/(pp+ap+pa+aa)
    #t2 = time.time()
    #if x==0: print("Topo invariant took = ", (t2-t1)/60. )
        
    ci = 2*np.log2(ti_val)
    #ci = 2*np.sign(pp)
        
        
    return ci
    
    
#@njit(parallel=True)
def sample_co_info(Lx,Ly,p,N_samples):

    # n_errors : number of sites with errors
    # N_samples : number of samples
    # p : bit-flip probability
    #Lx : length on x
    #Ly : length on y
    
    
    info = np.zeros(N_samples)
    
    num_qubits = 2*Lx*Ly
    
    rng = np.random.default_rng()
   

    for s in range(N_samples):
    
        #t1 = time.time()

        #random.shuffle(indices)
        #new_error = errors[indices]
        #print(bin_removed)
        #print(indices)
        
    
        new_error = rng.binomial(1, p, num_qubits)
        #print(new_error)
       
        co_info = topo_invariant(Lx,Ly,p,new_error)
        
        info[s] = co_info
        
        #t2 = time.time()
        #print("Time = ", (t2-t1)/60., "minutes")
        
    return info
    
def write_stats(stat_dict, h5F):
    for key in stat_dict:
        h5F[key] = stat_dict[key]
   
L = int(sys.argv[1])
#L = 3
Lx = L
Ly = L

#ps = np.linspace(1e-5,1e-3,10)
#ps = np.linspace(0.09,0.13,10)
ps = np.linspace(0.106,0.112,20) #for the paper
ps = ps[::2]
#print(ps)
#ps = np.array([0.1])


num_qubits = 2*Lx*Ly


    
N=1000000 # Samples
#N=1

#N = d*5000

print("Number of samples = ", N)
print("Number of physical qubits = ", num_qubits)



tol=1e-2
print("tolerance is = ", tol)

yerr = np.zeros(ps.shape)
y = np.zeros(ps.shape)

for i,p in enumerate(ps):

    print("idx, ps = ", i, " ", p)
    t1 = time.time()
    infos = sample_co_info(Lx,Ly,p,N)
    
    #print(info_sectors_eb)
    
    
    #prob_vec = compute_probability_vector(p,num_qubits)
    #print(np.sum(prob_vec))
    t2 = time.time()
    print("Total time is = ", (t2-t1)/60. , " minutes" )
   
    y[i] = np.mean(infos)
    #yerr[i] = stats.sem(infos)
    yerr[i] = simple_bootstrap(infos)
    
    print(y[i], yerr[i])
    


    
#Is = 4*np.log2(np.square(1-ps)+np.square(ps))+2
Is = 4*( ps*np.log2(ps)+(1-ps)*np.log2(1-ps))+2

print(Is)
print(y)
print(y_err)


#pyplot.errorbar(ps,y,yerr=yerr,fmt="-o",label="d={}".format(L))
    
#pyplot.plot(ps,Is,"-",label="actual value",color="orange")

#pyplot.axvline(x=np.sqrt(2)-1, color="grey",label="$\sqrt{2}-1$")
#pyplot.legend()

#pyplot.show()





ret={}

#output_file = "coherent_information_fermion_L{}_.h5".format(L)

#output_file = "coherent_information_fermion_L{}_second.h5".format(L)

output_file = "coherent_information_fermion_L{}_p_small.h5".format(L)

oF=h5py.File(output_file,'w')

ret["ps"] = ps
ret["ci"] =  y
ret["ci_eb"] =  yerr

write_stats(ret,oF)
oF.close()

