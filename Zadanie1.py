import math
import numpy as np
import matplotlib.pyplot as plt

f = lambda y : y*y*y-1
df = lambda y : 3*y*y
d2f = lambda y : 6*y

grid_size = 3000
y1,x1 = np.ogrid[-1.5:1.5:grid_size*1j, -1.5:1.5:grid_size*1j ]
z1=x1+y1*1j

def halley_iteration (z, n):
    r = z
    for i in range(1,n):
        t0 = f(r)
        t1 = df(r)
        t2 = d2f(r)
        r = r- 2*t0*t1/(2*t1*t1-t0*t2)
    return r
plt.figure(figsize=(10,10))
plt.imshow(np.angle(halley_iteration(z1,10)))
plt.show()
