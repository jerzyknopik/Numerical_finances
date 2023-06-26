import math
import numpy as np


def min_find(f,a,b,T0,Tmin,q):
    T=T0
    x=np.random.uniform(a,b)
    y=x
    acc=0
    while T>Tmin:
        y=x+np.random.normal(0,(b-a)*T/10)
        if y<a:
            y = 2*a-y
        if y>b:
            y = 2*b-y
        delta=f(y)-f(x)
        if(delta>0):
            if(np.random.uniform(low=0,high=1) < math.exp(-delta/T)):
                x=y
        else:
             x=y
        T=q*T
        #T=T0/math.log(acc+2)
        #acc+=1
    return x
g= lambda x: x*x
h= lambda x: 4*math.sin(x)/(math.cosh(0.5*x-3))

# Wykonanie algorytmu trwa u mnie ok. 30 sekund,
# ale jest to konieczne żeby znalazł on globalne
# minimum, a nie przypadkowe lokalne.
print(min_find(g,-1.,1.,1,1e-5,1-5e-6))
print(min_find(h,0.,20,1,1e-5,1-5e-6))

