import numpy as np
"""
This file includes the noise models for the ML-Decoder.
The noise here is the results of propagating the noise through the circuit until we get a phenemological noise model.
"""

"""
Helper functions
"""
def odd_occ_vec(n):
    """
    return a list of all odd n bit occupation vectors 

    :param n: 2^n binary numbers 
    """
    # complex way of getting implementing: bit_k(n) = (n >> k) & 1 
    # we are getting LSB (least significant bit first but this is ok, because we want all)
    ns = np.arange(0,2**(n),)
    occ_vec = ((ns[:, None] >> np.arange(n)) & 1).astype(np.uint8) 
    # now sort out all non odd values (reduce the length by factor 2)
    odd_occ_vec = occ_vec[np.sum(occ_vec,axis=1) % 2 == 1]
    return odd_occ_vec 

def prob_combined_flip_channels(ps):
    """
    This function combines the probabilites of many independent bit/phase flip channels
    into the probability of a concatenated/combined channel.
    """
    # 1. step generate all possible occupation vectors 2^(n_ps)
    # only the odd ones contribute to the possibility of bit flips
    ps = np.array(ps)
    ns = odd_occ_vec(len(ps))
    """
    pc = 0
    for n in ns:
        p_temp = 1
        for i,p in enumerate(ps):
            p_temp *= (1-p)**(1-n[i]) * p ** n[i]
        pc += p_temp
    """
    # optimized why of writing the above
    pc = np.sum(np.prod(ps**ns*(1-ps)**(1-ns),axis=1))
    return pc

def prob_2_depo_channel(p1,p2):
    return p1 + (1-4/3*p1)*p2

def prob_combined_depo_channels(pd):
    """
    Combine the probabilities of multiple concatenated depo1 channels to one cnot channel.
    
    :param pd: Description
    """
    # use recursive function
    if len(pd)==2:
        return prob_2_depo_channel(*pd) 
    else:
        pd = [
            *pd[:-2],
            prob_2_depo_channel(pd[-2],pd[-1])
        ] 
        return prob_combined_depo_channels(pd)


"""
Correlated noise models
"""

def correlated_unified_channel(pd,px,pz):
    pxt = 1/3*pd + (1-4/3*pd)* px * (1-pz)
    pzt = 1/3*pd + (1-4/3*pd)* (1-px) * pz
    pyt = 1/3*pd + (1-4/3*pd)* px * pz 
    return pxt, pyt, pzt 

"""
Uncorrelated noise models
"""