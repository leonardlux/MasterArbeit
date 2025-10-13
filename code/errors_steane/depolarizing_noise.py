# In this file I check my values for the probability assuming error propagation to the end of the ciruit, assumin depolarizing errors.

import numpy as np
import matplotlib.pyplot as plt 


def generate_function(a,b,c,d):
    # We should always expect 16 terms for our model!
    if a + b + c + d != 16:
        raise ValueError 
    def func(p):
        return a*(1-p)**3 + b/3*(1-p)**2*p + c/3**2*(1-p)*p**2 + d/3**3*p**3 
    return func

p_1 = generate_function(1,2,5,8)
p_x = generate_function(0,3,6,7)
p_y = generate_function(0,1,10,5)
p_z = generate_function(0,3,6,7)

ps = (p_1,p_x,p_y,p_z) 
titles = ("1","x","y","z")

p = np.linspace(0,1)

for i in range(len(ps)):
    plt.plot(p,ps[i](p),label=titles[i])
plt.plot(p, np.sum([pi(p) for pi in ps],axis=0), label="sum")

plt.title("Depolarizing Noise")
plt.legend()
plt.show()