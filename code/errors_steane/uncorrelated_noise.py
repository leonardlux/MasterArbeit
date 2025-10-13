# some code for uncorrelated errors propageted to the end of steane only on data qubit
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm

def gen_el(e1,en1,e2,en2,n=1):
    # e1 = exponent of p_1
    # en1 = exponent of (1-p_1)
    if e1 + e2 + en1 + en2 !=6:
        raise ValueError
    return lambda p1, p2: n * p1**e1 * (1-p1)**en1 * p2**e2 * (1-p2)**en2  

def gen_func(eliste):
    temp = [gen_el(*arg) for arg in eliste]

    return lambda p1, p2 : np.sum([tmp(p1,p2) for tmp in temp],axis=0) 

prob_1_list = (
    (0,3,0,3),
    (1,2,0,3),
    (0,3,1,2),
    (2,1,0,3),
    (2,1,1,2),
    (1,2,1,2),
    (1,2,2,1),
    (0,3,2,1),
    (3,0,0,3),
    (3,0,1,2),
    (3,0,2,1),
    (2,1,2,1),
    (3,0,3,0),
    (2,1,3,0),
    (1,2,3,0),
    (0,3,3,0)
)

prob_x_list = (
    (1,2,0,3,2),
    (2,1,0,3,2),
    (2,1,3,0,2),
    (1,2,3,0,2),
    (1,2,1,2,2),
    (2,1,1,2,2),
    (2,1,2,1,2),
    (1,2,2,1,2)
)

prob_y_list = (
    (1,2,1,2,4),
    (2,1,1,2,4),
    (2,1,2,1,4),
    (1,2,2,1,4)
)

prob_z_list = (
    (0,3,1,2,2),
    (0,3,2,1,2),
    (3,0,2,1,2),
    (3,0,1,2,2),
    (1,2,1,2,2),
    (1,2,2,1,2),
    (2,1,2,1,2),
    (2,1,1,2,2)
)
if 0:
    prob_1_func = gen_func(prob_1_list)
    prob_x_func = gen_func(prob_x_list)
    prob_y_func = gen_func(prob_y_list)
    prob_z_func = gen_func(prob_z_list)
elif 1:
    prob_1_func = lambda p1,p2 : (1-p1)*(1-p2) 
    prob_x_func = lambda p1,p2 : p1*(1-p2) 
    prob_y_func = lambda p1,p2 : p1*p2
    prob_z_func = lambda p1,p2 : p2*(1-p1)
else:
    prob_1_func = lambda p1,p2 : (1-p1)**2*(1-p2)**2 + p1**2 * (1-p2)**2 + (1-p1)**2 *p2**2 + p1**2 * p2**2
    prob_x_func = lambda p1,p2 : p1*(1-p1)*2*((1-p2)**2+p2**2) 
    prob_y_func = lambda p1,p2 : 4*(1-p1)*(1-p2)*p1*p2
    prob_z_func = lambda p1,p2 : p2*(1-p2)*2*((1-p1)**2+p1**2) 

    

def visu(x,y,z,title=""):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title(title)
    ax.plot_surface(x, y, z, vmin=z.min() * 2, cmap=cm.Blues)
    plt.show()

def visualize(func,title=""):
    p1 = np.linspace(0,1)
    p2 = np.linspace(0,1)
    p1,p2 = np.meshgrid(p1,p2)
    p = func(p1,p2)
    visu(p1,p2,p,title)

visualize(prob_1_func, "P1")
visualize(prob_x_func, "Px")
visualize(prob_z_func, "Pz")
visualize(prob_y_func, "Py")

def visualize_product(func1,func2,title=""):
    p1 = np.linspace(0,1)
    p2 = np.linspace(0,1)
    p1,p2 = np.meshgrid(p1,p2)
    p = func1(p1,p2) * func2(p1,p2)

    visu(p1,p2,p,title)

visualize_product(prob_x_func,prob_z_func,title="px * pz")

def visualize_sum(*funcs):
    p1 = np.linspace(0,1)
    p2 = np.linspace(0,1)
    p1,p2 = np.meshgrid(p1,p2)
    p = np.sum([func(p1,p2) for func in funcs],axis=0) 
    print(p)

visualize_sum(
    prob_1_func,
    prob_x_func,
    prob_y_func,
    prob_z_func
)

# and now p1=p2

p = np.linspace(0,1)
ps = (
    prob_1_func,
    prob_x_func,
    prob_y_func,
    prob_z_func,
)
titles = (
    "p1",
    "px",
    "py",
    "pz"
)


for i in range(len(ps)):
    plt.plot(p,ps[i](p,p),label=titles[i])
plt.plot(p, np.sum([pi(p,p) for pi in ps],axis=0), label="sum")
plt.title("p1=p2")
plt.legend()
plt.show()

