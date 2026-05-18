import numpy as np 
import matplotlib.pyplot as plt

from tools.error_propagation import corr_eff_noise, prob_combined_flip_channels, prob_combined_depo_channels

# Compositions
def pcz(p):
    return p

def pcx(p):
    return p**3 *(64/45) - p**2 *(28/9) + p*(11/5)

def pcd(p):
    return p**3 * (64/45) - p**2 * (52/15) + p* (14/5)

# Syndrome Channel

def ptij(pd,px,pz,i,j):
    if i==0:
        px = 1-px 
    if j==0:
        pz = 1-pz 
    return 1/3*pd + (1-4/3*pd)*px*pz

# Expaned syndrome channel:

def pty(p):
    # x=p
    return -(16384 * p**7)/6075 + (75776 * p**6)/6075 - (9664 * p**5)/405 + (15664 * p**4)/675 - (7324 * p**3)/675 + (47 * p**2)/45 + (14 * p)/15

def ptx(p):
    return (16384 * p**7)/6075 - (2048 * p**6)/135 + (220736 * p**5)/6075 - (95312 * p**4)/2025 + (7876 * p**3)/225 - (367 * p**2)/25 + (47 * p)/15

def ptz(p):
    return (16384 * p**7)/6075 - (75776 * p**6)/6075 + (9664 * p**5)/405 - (5648 * p**4)/225 + (11084 * p**3)/675 - (319 * p**2)/45 + (29 * p)/15

def pti(p):
    return -(16384 * p**7)/6075 + (2048 * p**6)/135 - (220736 * p**5)/6075 + (99152 * p**4)/2025 - (27388 * p**3)/675 + (4663 * p**2)/225 - 6 * p + 1


# Factorizing test
def pxc_ps(p_sp_plus, p_D2_plus, p_m_plus):
    a = (2/3) * p_sp_plus
    b = (2/3) * (4/5) * p_D2_plus
    c = p_m_plus
    return 4*(a*b*c) - 2*(a*b + a*c + b*c) + (a + b + c)

def pdc_ps(p_sp_psi, p_sp_0, p_D2_0):
    a = p_sp_psi
    b = p_sp_0
    c = (4/5) * p_D2_0
    return (a + b + c) - (4/3)*(a*b + a*c + b*c) + (16/9)*(a*b*c)

if False:
    # composite channels
    ps = np.linspace(0,1)
    plt.figure()
    plt.title("Composition")
    plt.plot(ps,pcz(ps),label="p_z")
    plt.plot(ps,pcx(ps),label="p_x")
    plt.plot(ps,pcd(ps),label="p_D")
    # plt.plot(ps,[prob_combined_flip_channels([p,2/3*p,2/3*4/5*p]) for p in ps], label="python p_x")
    # plt.plot(ps,[prob_combined_depo_channels([p,p,4/5*p]) for p in ps], label="python p_d")

    plt.legend()
    plt.xlabel("p")
    plt.ylabel("prob. of composition")
    plt.show()

if False:
    # Syndrome Channel
    ps = np.linspace(0,1.5,1000)
    plt.figure()
    plt.title("Syndrome channel")

    # plt.plot(ps,ptij(pcd(ps),pcx(ps),pcz(ps),0,0),label="I")
    plt.plot(ps,pti(ps),label="I_exp")
    # # plt.plot(ps,ptij(pcd(ps),pcx(ps),pcz(ps),0,0)-pti(ps),label="diff I")

    # plt.plot(ps,ptij(pcd(ps),pcx(ps),pcz(ps),1,0),label="X")
    plt.plot(ps,ptx(ps),label="X_exp")
    # # plt.plot(ps,ptij(pcd(ps),pcx(ps),pcz(ps),1,0)-ptx(ps),label="diff X")

    # plt.plot(ps,ptij(pcd(ps),pcx(ps),pcz(ps),0,1),label="Z")
    plt.plot(ps,ptz(ps),label="Z_exp")
    # # plt.plot(ps,ptij(pcd(ps),pcx(ps),pcz(ps),0,1)-ptz(ps),label="Z diff")

    # plt.plot(ps,ptij(pcd(ps),pcx(ps),pcz(ps),1,1),label="Y")
    plt.plot(ps,pty(ps),label="Y_exp")
    # plt.plot(ps,ptij(pcd(ps),pcx(ps),pcz(ps),1,1)-pty(ps),label="Y diff")

    # One of my implementations is wrong!
    # res = np.array([corr_eff_noise(p) for p in ps])
    # plt.plot(ps,res[:,0]-ptx(ps),label="X python")
    # plt.plot(ps,res[:,1]-pty(ps),label="Y python")
    # plt.plot(ps,res[:,2]-ptz(ps),label="Z python")
    # plt.plot(ps,res[:,3]-pti(ps),label="I python")

    # sum
    # plt.plot(ps, res[:,0]+res[:,1]+res[:,2]+res[:,3]-1, label="python")
    # plt.plot(ps,pti(ps)+ptx(ps)+pty(ps)+ptz(ps)-1, label="exp") 

    plt.xlabel("p")
    plt.ylabel("prob. of error")
    plt.legend()
    plt.show()


# Correlated channels

def p_corr(i_s,i_psi, 
           p_sp_psi, p_sp_0, p_sp_p,
           p_d2_0, p_d2_p,
           p_m0, p_mp,
           ):
        """
        return probabilities for p_{i_s,i_psi}, so for the correlated error channel at the end of the circuit. 
        """
        # 1_s
        if i_s == 0:
            if i_psi == 0:
                # 1_s 1_psi
                return 1 - p_sp_psi - p_sp_0 - p_sp_p - 14/15* (p_d2_0 + p_d2_p) - p_m0 - p_mp 
            elif i_psi == 1:
                # 1_s X_psi
                return 2/15* p_d2_p
            elif i_psi == 2:
                # 1_s Z_psi
                return 1/3* p_sp_p + 2/15 * (p_d2_0 + p_d2_p) 
            elif i_psi == 3:
                # 1_s Y_psi 
                return 2/15*p_d2_p
        # X_s
        elif i_s == 1:
            if i_psi == 0:
                # X_s 1_psi
                return 1/3 * p_sp_p + p_m0 + 2/15*(p_d2_p)
            elif i_psi == 1:
                # X_s X_psi
                return 1/3 * (p_sp_psi + p_sp_0) + 2/15*(p_d2_0 + p_d2_p)
            elif i_psi == 2:
                # X_s Z_psi
                return 1/3 * p_sp_p + 2/15*(p_d2_p)
            elif i_psi == 3:
                # X_s Y_psi
                return 2/15*(p_d2_0 + p_d2_p)
        # Z_s 
        elif i_s == 2:
            if i_psi == 0:
                # Z_s 1_psi
                return 1/3 * p_sp_0 + p_mp + 2/15*(p_d2_0)
            elif i_psi == 1:
                # Z_s X_psi
                return 0 # O(p^2)
            elif i_psi == 2:
                # Z_s Z_psi
                return 1/3 * p_sp_psi + 2/15*(p_d2_0)
            elif i_psi == 3:
                # Z_s Y_psi
                return 0 # O(p^2)
        # Y_s
        elif i_s == 3:
            if i_psi == 0:
                # Y_s 1_psi
                return 0 # O(p^2)
            elif i_psi == 1:
                # Y_s X_psi
                return 1/3 * p_sp_0 + 2/15 * p_d2_0
            elif i_psi == 2:
                # Y_s Z_psi
                return 0 # O(p^2)
            elif i_psi == 3:
                # Y_s Y_psi
                return 1/3* p_sp_psi + 2/15* p_d2_0

def p_corr_simplified(i_s, i_psi, p):
    """ 
    just all p equal 
    """
    return p_corr(i_s, i_psi, p, p, p, p, p, p, p,)
