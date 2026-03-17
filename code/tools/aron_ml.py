import numpy as np
import numba 


@numba.njit()
def error_converter(d,E):
    """
    Convert error string to form which is more natural for the mld algorithm
    """
    f=np.zeros((2,d,d))
    for i in range(d-1):
        for j in range(d):
            # 0 is horzitontal
            f[0,j,i] = E[i*(2*d-1)+j]
        for j in range(d-1):
            # 1 is vertical
            f[1,j,i] = E[i*(2*d-1)+d+j]
    for j in range(d):
        f[0,j,d-1] = E[(d-1)*(2*d-1)+j]
    return f

#Optimal algorithm for maximum likmelihood decoding
@numba.njit
def initialize_M0(d):
    """
    Creation of the initial covariance metrix M0. This is the covariance matrix of the initial gaussian state |psi_e>.
    """
    M0=np.zeros((2*d,2*d))
    for i in range(d-1):
        M0[2*i+1,2*i+2]=1
        M0[2*i+2,2*i+1]=-1
    M0[0,2*d-1]=1
    M0[2*d-1,0]=-1
    return M0

@numba.njit
def simulate_H_columns(M,j,log_gamma,f,p,d):
    """
    Algorithm for simulating the action of Gaussian operator H_j.
    """
    w=np.zeros(d)
    t=np.zeros(d)
    s=np.zeros(d)
    A=np.zeros((2*d,2*d))
    B=np.zeros((2*d,2*d))
    for i in range(d):
        if f[0,j,i]==1:
            w[i]=(1-p)/p
        else:
            w[i]=p/(1-p)
        log_gamma=log_gamma+np.log((1+w[i]**2)/2)
        t[i]=(1-w[i]**2)/(1+w[i]**2)
        s[i]=2*w[i]/(1+w[i]**2)
        A[2*i,2*i+1]=t[i]
        A[2*i+1,2*i]=-t[i]
        B[2*i,2*i]=s[i]
        B[2*i+1,2*i+1]=s[i]
    log_gamma=log_gamma+np.log(np.sqrt(np.linalg.det(M+A)))
    log_gamma=log_gamma+np.log(((1-p)**(d-np.sum(f[0,j,:]))*p**np.sum(f[0,j,:]))**2)
    M=A-(B@np.linalg.inv(M+A)@B)
    return M,log_gamma

@numba.njit
def simulate_V_columns(M,j,log_gamma,f,p,d):
    """
    Algorithm for simulating the action of Gaussian operator V_j.
    """
    w=np.zeros(d-1)
    t=np.zeros(d-1)
    s=np.zeros(d-1)
    A=np.zeros((2*d,2*d))
    B=np.zeros((2*d,2*d))
    B[0,0]=1
    B[2*d-1,2*d-1]=1
    for i in range(d-1):
        if f[1,j,i]==1:
            w[i]=(1-p)/p
        else:
            w[i]=p/(1-p)
        log_gamma=log_gamma+np.log(1+w[i]**2)
        t[i]=2*w[i]/(1+w[i]**2)
        s[i]=(1-w[i]**2)/(1+w[i]**2)
        A[2*i+1,2*i+2]=t[i]
        A[2*i+2,2*i+1]=-t[i]
        B[2*i+1,2*i+1]=s[i]
        B[2*i+2,2*i+2]=s[i]
    log_gamma=log_gamma+np.log(np.sqrt(np.linalg.det(M+A)))
    log_gamma=log_gamma+np.log(((1-p)**(d-1-np.sum(f[1,j,:]))*p**np.sum(f[1,j,:]))**2)
    M=A-(B@np.linalg.inv(M+A)@B)
    return M,log_gamma

@numba.njit
def simulation_mld(p,d,f):
    """
    Calculation of ln(Z({w_E})) from an inital error string, and the probability of errors.
    """
    M0=initialize_M0(d)
    M=M0.copy()
    log_gamma=np.log(2**(d-1))
    for j in range(d-1):
        M, log_gamma=simulate_H_columns(M,j,log_gamma,f,p,d)
        M, log_gamma=simulate_V_columns(M,j,log_gamma,f,p,d)  
    M, log_gamma=simulate_H_columns(M,d-1,log_gamma,f,p,d)
    return 1/2*log_gamma-1/2*np.log(2)+np.log(np.linalg.det(M+M0)**(1/4))
