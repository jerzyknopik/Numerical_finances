#Jerzy Knopik 26.01.2023

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as lg
import scipy.optimize as opt



N = 100
x = np.linspace(0, 10, N)
h = 10/(N-1)
m0=math.floor(1/h)

def constant_step_example(nsteps = 1000, every=100, gamma=0.49*h**2):
    u = x/10
    results = [u]
    plot1=[u[m0]]

    for k in range(nsteps):
        v = u[1:-1]    
        u[1:-1] += gamma*(np.diff(u, 2)/(h*h) - 2*v*(v**2-1))
        plot1.append(u[m0])
        if k%every==0: results.append(np.copy(u))
    return [np.array(results).reshape(-1, x.size).T,plot1]

rt= np.linspace(0, 1002, 1001)
rt*=0.49*h**2
r = constant_step_example(every=20)
def fit(t, a, b,omega):
    return a+b*np.exp(-omega*t)
fitparams=opt.curve_fit(fit,rt[400:-1],r[1][400:-1])
fittedf=fit(rt,fitparams[0][0],fitparams[0][1],fitparams[0][2])
print(fitparams)
profiles=np.array([u-r[0][-1] for u in r[0]])
profiles=np.array([u/lg.norm(u) for u in profiles])
#a = plt.plot(x, r[0], alpha=0.3)
b = plt.plot(rt,r[1])
c = plt.plot(rt,fittedf)
plt.show()
d = plt.plot(x,profiles, alpha=0.3)
plt.show()
