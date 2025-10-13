import numpy as np
import sys
import os
import stim

from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
import scipy.sparse as spr
import time
import h5py
import itertools
#import numba
#from numba import njit, prange, jit
#import numba_sparse  # The import generates Numba overloads for special
#import numba_scipy

had = stim.Tableau.from_named_gate("H")
cx = stim.Tableau.from_named_gate("CNOT")

Xgate = stim.Tableau.from_named_gate("X")
Ygate = stim.Tableau.from_named_gate("Y")
Zgate = stim.Tableau.from_named_gate("Z")
Igate = stim.Tableau.from_named_gate("I")

one_qubit_generators = [Xgate,Ygate,Zgate]



def PauliOps(L,i,pauli_idx):
    ''' Spin Opertaors at each site '''
    sx = spr.csr_matrix(np.array([[0.,1.0],[1.0,0.]]))
    sy = spr.csr_matrix(np.array([[0.,-1.0j],[1.0j,0.]]))
    sz = spr.csr_matrix(np.array([[1.0,0.],[0.,-1.0]]))
    #Sxs = []
    #Sys = []
    #Szs = []
   
    leftdim=2**i
    rightdim=2**(L-i-1)

    if pauli_idx==0 : BigPauli = spr.kron(spr.kron(spr.eye(leftdim),sx),spr.eye(rightdim),'csr')
    elif pauli_idx==1 : BigPauli = spr.kron(spr.kron(spr.eye(leftdim),sy),spr.eye(rightdim),'csr')
    elif pauli_idx==2: BigPauli = spr.kron(spr.kron(spr.eye(leftdim),sz),spr.eye(rightdim),'csr')
    #Sxs.append(Sx)
    #Sys.append(Sy)
    #Szs.append(Sz)

    return BigPauli

def stabilizer_string(list_indices,x_stab,N):

    if x_stab==True: Pauli="X"
    else: Pauli="Z"
    
    string_pauli = ""
    for x in range(N):
        if x in list_indices: string_pauli += Pauli
        else: string_pauli += "I"
        
    return string_pauli


clifford_gates = [had,cx]

s1x = [0,3,5,6]
s2x = [1,3,4,7]
s3x = [2,4,5,8]

s4x = [6,9,11,12]
s5x = [7,9,10,13]
s6x = [8,10,11,14]

s7x = [0,12,15,17]
s8x = [1,13,15,16]


#zl1 = [1,7,13]
#zl1 = [9,10,11]

#z stabilizers
s1z = [0,1,3,15]
s2z = [1,2,4,16]
s3z = [3,6,7,9]

s4z = [4,7,8,10]
s5z = [5,6,8,11]
s6z = [9,12,13,15]


s7z = [10,13,14,16]
s8z = [11,12,14,17]



zl1 = [1,7,13]
zl2 = [9,10,11]

xl1 = [3,9,15]
xl2 = [6,7,8]


x_stab = [s1x,s2x,s3x,s4x,s5x,s6x,s7x,s8x]
z_stab = [s1z,s2z,s3z,s4z,s5z,s6z,s7z,s8z]
zl_list = [zl1,zl2]
xl_list = [xl1,xl2]

control = [0, 2, 4, 5, 6, 8, 10, 15]
target = [1, 3, 7, 9, 11, 12, 13, 14, 16, 17]

connections = [[0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1],[0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,1],[0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,1,0,1,0,1,1,0,0,0],[0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0]]



print("Error propagation")

class gate:
    def __init__(self,op_idx,control,target):
        self.op=clifford_gates[op_idx]
        self.target=target
        self.control=control
        if op_idx==0:
            self.single_qubit_gate=True
        else:
            self.single_qubit_gate=False
            
    def get_parameters(self):
        qubits = [self.control,self.target]
        sqg = self.single_qubit_gate
        
        return self.op, qubits, sqg

def gates_logical_zero(controls,targets,connections):

    #t = stim.Tableau(7)
    list_gates = []
    
    
    #pivotal qubits 0,4,6,12,15
    
    for x in controls: list_gates.append(gate(0,-1,x))
    
    for i,x in enumerate(controls):
        connec = connections[i]
        for j,y in enumerate(connec):
            if y==1: list_gates.append(gate(1,x,j))
            
    return list_gates
    
def build_tableu(gate_list):
    N=18
    t=stim.Tableau(N)
    #t = stim.TableauSimulator(17)
    for ga in gate_list:
        op, control_target, sqg = ga.get_parameters()
        #print(op,control_target)
        if sqg==True:
            t.append(op,[control_target[1]])
        else:
            t.append(op,control_target)
            
    return t.inverse()
    
    
def propagate_error(pauli_idx,site_idx,unitary):
    "Propagate back errors"
    #err_t = stim.Tableau(7)
    
    #err_t.append(one_qubit_generators[pauli_idx],site_idx)
    N=18
    #err = err_t.to_pauli_string()
    list_indices = [0]*N
    list_indices[site_idx] = pauli_idx+1
    err = stim.PauliString(list_indices)
    #print(err)
    #unitary = build_tableu(new_gate_list)
    err_prop = err.after(unitary,targets=range(N))
                
    return err_prop
    
def propagate_error_z(unitary):
    "Propagate back errors"
    #err_t = stim.Tableau(7)
    
    #err_t.append(one_qubit_generators[pauli_idx],site_idx)
    
    #err = err_t.to_pauli_string()
    
    N=18
    for x in range(N):
        list_indices = [0]*N
        list_indices[x] = 3
        err = stim.PauliString(list_indices)
        #print(err)
        #unitary = build_tableu(new_gate_list)
        err_prop = err.before(unitary,targets=range(N))
        
        print(err_prop)
                
    return err_prop


def check_logical_one(control,target,connections):
    
    N=18
    s = stim.TableauSimulator()
    s.set_num_qubits(N)
    
    gatelist =  gates_logical_zero(control,target,connections)
    unitary = build_tableu(gatelist)
    
   
      
    for g in gatelist:
        op, control_target, sqg = g.get_parameters()
        control = control_target[0]
        target = control_target[1]
            
        if sqg==True :
            s.do_tableau(op,[target])
                
        else:
            s.do_tableau(op,control_target)
            
        
    energy = 0.0
    #zl = "IIIIZIIIZIIIZIIZZ"
    
    list_stabilizers = []
    list_logicals = []
    
    for stab in z_stab:
        stab_string = stabilizer_string(stab,False,18)
        list_stabilizers.append(stab_string)
            

    for stab in x_stab:
        stab_string = stabilizer_string(stab,True,18)
        list_stabilizers.append(stab_string)
    
    for stab in zl_list:
        stab_string = stabilizer_string(stab,False,18)
        list_logicals.append(stab_string)
    
    
    for i,stab in enumerate(list_stabilizers):
        obs = stim.PauliString(stab)
        energy -= s.peek_observable_expectation(obs)
        
    for i,stab in enumerate(list_logicals):
        obs = stim.PauliString(stab)
        print("logical = ", s.peek_observable_expectation(obs))
    
    #stab_string = stabilizer_string(zl,False,25)
    #list_stabilizers.append(stab_string)
    
    #obs = stim.PauliString(stab_string)
    #energy_zl = s.peek_observable_expectation(obs)
        
        
    return energy
    
def sites_logical_information(control,target,connections):

    N=18
    s = stim.TableauSimulator()
    s.set_num_qubits(N)
    
    gatelist =  gates_logical_zero(control,target,connections)
    unitary = build_tableu(gatelist)
    
    
    
    #s.do_tableau(unitary.inverse(),17)
    #s.do_tableau(Xgate,[1])
    #s.do_tableau(Xgate,[5])
    #s.do_tableau(Xgate,[10])
    #s.do_tableau(Xgate,[11])
    #s.do_tableau(Xgate,[12])
    
      
    for g in gatelist:
        op, control_target, sqg = g.get_parameters()
        control = control_target[0]
        target = control_target[1]
            
        if sqg==True :
            s.do_tableau(op,[target])
                
        else:
            s.do_tableau(op,control_target)
            
    #apply logical x
    
    list_indices = []
    
    for xl in xl_list:
        print("logical operator")
        #xl = [20,21,22,23,24]
        #s.do_tableau(Xgate,xl)
        
        s2  = s.copy()
        
        for i in xl: s2.do_tableau(Xgate,[i])
        
        list_x_indices = []
        for x in range(N):
            obs = propagate_error(2,x,unitary.inverse())
            #obs = stim.PauliString()
            #print(obs)
            expc_value = s2.peek_observable_expectation(obs)
            print("index = ", x)
            print("value = ", expc_value)
            print("")
        
            if expc_value==-1 : list_x_indices.append(x)
            
        list_indices.append(list_x_indices)
        
    return list_indices
        
        
    


def prepare_noisy_bell_steane(p,list_x_indices,N,k):
    "Noisy bell with back propagated errors"
    "zzz logical: 000000000000000|000"
    "zzo logical: 000001100100000|001"
    "zoz logical: 000000110010000|010"
    "ozz logical: 000001011000000|100"
    
    
    indices = []
    
    #create logical bit-strings
    
    x_log = np.zeros((k,N),dtype=int)
    
    for i in range(k):
        for x in list_x_indices[i]:
            x_log[i,x] = 1
            
    print(x_log)
    
    vec_convert = 1<<np.arange(N+k)
    vec_convert = vec_convert[::-1]
    
    for idx in range(2**k):
        #index_string =  "0"*(N+k)
        zeros = np.zeros(N+k,dtype=int)
        bin_array = np.array([int(x) for x in bin(idx)[2:].zfill(k)])
        #print(bin_array)
        #bin_array = np.fromstring(bin_array, dtype=int, sep='')
        #print(bin_array)
        #bin_array = np.unpackbits(idx)
        print(bin_array)
        
        data_qubit = np.zeros(N,dtype=int)
        
        for j in range(k):
            if bin_array[j]==1 : data_qubit += x_log[j,:]
        
        data_qubit = np.mod(data_qubit,2)
        reference_qubit = np.copy(bin_array)
        #reference_qubit[idx] = 1
        print(data_qubit)
        
        bit_string = np.concatenate((data_qubit,reference_qubit),axis=None)
        print(bit_string)
        
        index = np.dot(vec_convert,bit_string)
        print(index)
        indices.append(index)
    
    
    print(indices)
    pairs = itertools.product(indices, repeat=2)
    
    #create_matrix_elements
    
    row = np.zeros(2**(2*k))
    col = np.zeros(2**(2*k))
    data = np.zeros(2**(2*k))
    
    for idx,ele in enumerate(pairs):
        row[idx] = ele[0]
        col[idx] = ele[1]
        data[idx] = 1/(2**k)
    
    #N=25
    print(row)
    print(col)
    
    
    rho = coo_matrix((data,(row,col)), shape=(2**(N+k),2**(N+k)))
    
    #rho_temp = rho.tocsr()
    rho = rho.tocsr()
    
    
    #Xs, Zs = PauliOps(N+1)
    #print("Hi")
    #paulis = [Xs,Ys,Zs]
    #apply errors
    #gatelist =  gates_logical_zero()
    gatelist =  gates_logical_zero(control,target,connections)
    unitary = build_tableu(gatelist)
    #rho = rho.toarray()
    
    #x_pauli = PauliOps(N+1,1,0)
    #z_pauli = PauliOps(N+1,1,2)
    
    
    for x in range(N):
        print("x = ", x)
        rho_temp1 = rho.copy()
        
        rho = (1-p)*rho
        
        t1 = time.time()
        
        #rho_temp2 = rho_temp1.copy()
        err_string = propagate_error(0,x,unitary)
        #print("prop = ", err_string)
        #err_wo_zeros = [i for i in err_string if i != 0]
        
        
        for idx,e in enumerate(list(err_string)):
            
            #x_pauli = PauliOps(N+1,idx,0)
            #z_pauli = PauliOps(N+1,idx,2)
            #print(idx,e)
            if e==2:
                #rho_temp2 = (Xs[idx]*Zs[idx])*rho_temp2*(Zs[idx]*Xs[idx])
                x_pauli = PauliOps(N+k,idx,0)
                z_pauli = PauliOps(N+k,idx,2)
                rho_temp1 =  (x_pauli*z_pauli) * rho_temp1 * (z_pauli*x_pauli)
                #print("Y e = ", e)
            elif e==1:
                x_pauli = PauliOps(N+k,idx,0)
                rho_temp1 = x_pauli * rho_temp1 * x_pauli
                #rho_temp2 = Xs[idx] * rho_temp2 * Xs[idx]
            elif e==3:
                z_pauli = PauliOps(N+k,idx,2)
                rho_temp1 = z_pauli * rho_temp1 * z_pauli
                #rho_temp2 = Zs[idx] * rho_temp2 * Zs[idx]
                    
        rho = rho+rho_temp1*p
                   
        t2 = time.time()
        print("Error applied in ", (t2-t1)/60., " minutes")
                
        
    
        
    rho.eliminate_zeros()
    #print(rho+(-1)*rho.transpose())
    print(rho.count_nonzero())
    #rho = rho.tocoo()
    #print(rho)
    #epsilon = 1e-12
    #mask = np.abs(rho.data)>epsilon
    #print(rho.data.shape)
    #print(rho.data[np.abs(rho.data)<1e-14])
    #print(rho.data[np.abs(rho.data)<1e-14].shape)
    #print("cutoff = ", epsilon)
    #print(rho.trace())
    #print(rho.row)
    #print(rho.col)
   
    #print(rho.col==index_00)
    #print(rho.row==index_00)
    
    #print(rho.col == index_11)
    #print(rho.row == index_11)
    
    #mask = (rho.col==index_00) & (rho.row==index_11)
    #print(mask)
    #print(data[mask])
    #print(list_x_indices)
    
    return rho.todok()
    
def compute_coherent_information(rho_RQ,list_x_indices,N,k):

    #print(rho_RQ)
    indices = []
    indices_zero_reference = []
    
    #create logical indices
    
    x_log = np.zeros((k,N),dtype=int)
    
    for i in range(k):
        for x in list_x_indices[i]:
            x_log[i,x] = 1
            
    #print(x_log)
    
    vec_convert = 1<<np.arange(N+k)
    vec_convert = vec_convert[::-1]
    
    for idx in range(2**k):
        #index_string =  "0"*(N+k)
        zeros = np.zeros(N+k,dtype=int)
        bin_array = np.array([int(x) for x in bin(idx)[2:].zfill(k)])
        #print(bin_array)
        #bin_array = np.fromstring(bin_array, dtype=int, sep='')
        #print(bin_array)
        #bin_array = np.unpackbits(idx)
        #print(bin_array)
        
        data_qubit = np.zeros(N,dtype=int)
        
        for j in range(k):
            if bin_array[j]==1 : data_qubit += x_log[j,:]
        
        data_qubit = np.mod(data_qubit,2)
        reference_qubit = np.copy(bin_array)
        #reference_qubit[idx] = 1
        #print(data_qubit)
        
        bit_string = np.concatenate((data_qubit,reference_qubit),axis=None)
        bit_string_ref_zero = np.concatenate((data_qubit,np.zeros(k,dtype="int")),axis=None)
        #print(bit_string)
        
        index = np.dot(vec_convert,bit_string)
        index_ref_zero = np.dot(vec_convert,bit_string_ref_zero)
        #print(index)
        indices.append(index)
        indices_zero_reference.append(index_ref_zero)
        
    
    #create indices block density matrix
    
    local_pairs = [[] for i in range(2**k)]
    
    local_indices = []
    local_indices_zero = []
    
    
    
    for n in range(2**k):
        x1 = n
        x2 = np.left_shift(x1,k)
        num1 = np.bitwise_xor(x1,x2)
        
        local_indices.append(num1)
        local_indices_zero.append(x2)
    
    
    for m,idx2 in enumerate(local_indices_zero):
        #new_indices = []
        #[new_indices.append(np.bitwise_xor(old_idx,idx2)) for old_idx in local_indices]
        new_indices = np.bitwise_xor(np.array(local_indices),idx2)
        #idx_flipped =  np.bitwise_xor(idx1,idx2)
        new_pairs = itertools.product(new_indices, repeat=2)
        local_pairs[m].extend(new_pairs)
            
    #pairs = np.array(pairs)
        
    #print(pairs)
    
    #print(pairs.shape)
    
    #print(indices)

    #local_pairs = list(local_pairs)
    
    #print(index_00_old)
    #print(index_01_old)
    #print(index_10_old)
    #print(index_11_old)
    
    #vec_convert = 1<<np.arange(N+1)
    #vec_convert = vec_convert[::-1]
    #print(vec_convert)
    #print("number of nonzero = ", data.shape)
    
    I = 0
    #norm = 0
    for x in range(2**N):
        #print(100*x/2**N, "percentage")
        t1 = time.time()
        #array_x = (((x & (1 << np.arange(N+1))[::-1])) > 0).astype(int)
        #y = 2*x
        #array_x = (((y & (1 << np.arange(N+1))[::-1])) > 0)
        #print(array_x)
        #t1 = time.time()
        y = np.left_shift(x,k)
        #print(y)
        #y = x
        #new_indices = []
        
        #for idx in indices: new_indices.append(np.bitwise_xor(idx,y))
        new_indices = np.bitwise_xor(indices,y)
        
        #print(new_indices)
        #pairs = itertools.product(new_indices, repeat=2)
        #new_indices = np.array(new_indices)
        
        rho = np.zeros((4**k,4**k))
        
        #for idx1 in new_indices:
        for j,idx0 in enumerate(indices_zero_reference):
        
            update_global_indices = np.bitwise_xor(np.array(new_indices),idx0)
            update_local_indices = np.bitwise_xor(np.array(local_indices),local_indices_zero[j])
            
            
            
            global_pairs = list(itertools.product(update_global_indices, repeat=2))
            local_pairs = list(itertools.product(update_local_indices, repeat=2))
            
            #if x==0 and j==0: print(global_pairs)
            #if x==0 and j==0: print(local_pairs)
            
            for i,prs in enumerate(global_pairs):
                #print(prs)
                local_idx1 = local_pairs[i][0]
                local_idx2 = local_pairs[i][1]
                
                #local_idx1 = local_pairs[j][i][0]
                #local_idx2 = local_pairs[j][i][1]
                
                global_idx1 = prs[0]
                global_idx2 = prs[1]
                #if x==0:
                       
                    
                rho[local_idx1,local_idx2] = rho_RQ[global_idx1,global_idx2]
                #rho[local_idx2,local_idx1] = rho_RQ[idx_flipped,idx1]
            
                #i+=1
        
        #if np.linalg.norm(rho)>0: print(rho)
        
        
        
        #norm += rho[0,0]+rho[1,1]+rho[2,2]+rho[3,3]
        #rho = 0.5*rho
        #if np.linalg.norm(rho) > 0 : print(rho)
       
            
        s1 = np.linalg.eigvalsh(rho)
        s2  = np.copy(s1)
           
        s1  = s1[s1>1e-15]
        Srq = - np.sum(s1*np.log(s1))
    
        
        rhoQ = np.trace(np.reshape(rho,(2**k,2**k,2**k,2**k)),axis1=1,axis2=3)
        
        
        #print(rhoQ)
        #print(rhoQ.shape)
        s1 = np.linalg.eigvalsh(rhoQ)
        
        
        #if np.linalg.norm(rho)>0:
        #print(np.trace(rho))
        #print(np.trace(np.dot(rho,rho)))
        #print(rho)
        #print(s2)
        #print(s1)
        #print("")
        
        #values.extend(s1)
        #print(s1)
        s1  = s1[s1>1e-15]
        Sq = - np.sum(s1*np.log(s1))
        #print(s1)
        #print(Sq-Srq)
        I+= (Sq-Srq)/(k*np.log(2))
        
        if x==0:
            print(I)
            print(k*np.log(2))
            
        #list_indices.append(index_00)
        #list_indices.append(index_11)
        #list_indices.append(index_01)
        #list_indices.append(index_10)
        t2 = time.time()
        #print("Took = ", (t2-t1)/60., " minutes")
        
    #print("trace of rho = ", norm*0.5)
    
       
    return I/(2**k)
            
            
    
def write_stats(stat_dict, h5F):
    for key in stat_dict:
        h5F[key] = stat_dict[key]


#print(check_logical_one())

gatelist =  gates_logical_zero(control,target,connections)
unitary = build_tableu(gatelist)
err = propagate_error_z(unitary)
energy = check_logical_one(control,target,connections)
print(energy)

list_x_indices = sites_logical_information(control,target,connections)

print(list_x_indices)

N=18
k=2


'''
t1 = time.time()
rho = prepare_noisy_bell_steane(0.0,list_x_indices,N,k)
#rho_row, rho_col, rho_data = prepare_noisy_bell_steane(0.106,list_x_indices,N)
t2 = time.time()
print("rho computed in ", (t2-t1)/60., " minutes")

t1 = time.time()
ci = compute_coherent_information(rho,list_x_indices,N,k)
print("New coherent information = ", ci)
t2 = time.time()
print("coherent information got in ", (t2-t1)/60., " minutes")
'''

#ps = np.array([0.184])
#ps = np.linspace(0.184,0.191,20)
#print(ps[0:5])
#ps = ps[5:10]
#ps = ps[10:15]
ps = np.linspace(0.106,0.112,20)
#ps = np.array([0.189])
#ps  =np.array([0])
#ps = np.logspace(-6,0,20)
info_co = np.zeros(ps.shape[0])
#info_co_2 = np.zeros(ps.shape[0])


print("p vector = ", ps)

for i,p in enumerate(ps):

    print("p = ", p)
    
    t1 = time.time()
    #rho_row, rho_col, rho_data = prepare_noisy_bell_steane(p,list_x_indices,N)
    rho = prepare_noisy_bell_steane(p,list_x_indices,N,k)
    #print(rho_data.shape)
    print("rho done")
    #ci = compute_coherent_information(rho_row,rho_col,rho_data,N)
    ci = compute_coherent_information(rho,list_x_indices,N,k)
    t2 = time.time()
    #info_co[i,0] = coherent_information(rho_dep_1,1,1)/np.log(2)
    print("CI = ", ci)
    info_co[i] = ci
    
    print("coherent information got in ", (t2-t1)/60., " minutes")
    
        
    
print(info_co)



ret={}


output_file = "tori_code_coherent_information_bit_flip.h5"



oF=h5py.File(output_file,'w')

ret["ci_18"] = info_co
ret["ps"] = ps



write_stats(ret,oF)
oF.close()




