#Jerzy Knopik 25.11.2022

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

#EULER METHOD
def newton_method(x0,f,df,n):
    x=x0
    for i in range(1,n):
        x=x-f(x)/df(x)
    return x

def explicit_euler(t0,y0,f,h,tk):
    t=t0
    y=y0
    sol=[(t0, y0)]
    while(t<tk):
        y=y+h*f(t,y)
        t=t+h
        sol.append((t, y)) 
    return sol

def implicit_euler(t0,y0,f,df,h,tk):
    t=t0
    y=y0
    sol=[(t0,y0)]
    g=lambda x : x-h*f(x)-y
    dg=lambda x: 1-h*df(x)
    while(t<tk):
        y=newton_method(y,g,dg,10)
        t=t+h
        sol.append((t,y))
    return sol

fun1 = lambda t, x : -x*x
fun10= lambda x : -x*x
fun2 = lambda x : -2*x
explicit_sol=explicit_euler(0,1,fun1,0.01,10)
implicit_sol=implicit_euler(0,1,fun10,fun2,0.01,10)
plt.figure(figsize=(10,5))
plt.title("Problem 1")
plt.plot(*zip(*explicit_sol))
plt.plot(*zip(*implicit_sol))
plt.show()
#2-dim EULER

m=np.array([[-501,500],[500,-501]])

w,v = np.linalg.eig(m)
print("Problem 2")
print("eigenvalues:", w)
print("eigenvectors:", v)
def explicit_2d(t0,h,tk):
    z=np.array([1,0])
    t=t0
    sol=[(t0,z)]
    while(t<tk):
        z=z+np.matmul(m,z)*h
        t=t+h
        sol.append((t,z))
    return sol
sol2d= explicit_2d(0,1e-5,1)
plt.plot(*zip(*sol2d))
plt.title("Problem 2")
plt.show()
        
#SHOOTING METHOD
def shootingf(L,h,a):
    t=0
    y=0
    v=a
    while(t<L):
        v=v+2*y*(y*y-1)*h
        y=y+v*h
        t=t+h
    return (y-1)

def bisect(t0,t1,f):
    a=t0
    b=t1
    while((b-a)>1e-10):
        c=(a+b)/2
        fc=f(c)
        if(fc<0):
            a=c
        else:
            b=c
    return (a+b)/2
        
x0=1
x1=1+1e-4
ashooting=bisect(x0,x1,partial(shootingf,5,1e-4))
print("Problem 6")
print("Value of a for L=5:", ashooting)
# This is a good result in general. Using usual
#integrating methods is not sufficient to get closer to the
#result. One should restore to symplectic Runge-Kutta
    
#LOTKA-VOLTERRA

def lotka_volterra(a,b,c,d,x0,y0,t0,h,tk):
    f=lambda z: np.array([a*z[0]+b*z[0]*z[1],c*z[1]+d*z[0]*z[1]])
    vec=np.array([x0,y0])
    sol=[(t0,vec)]
    t=t0
    while(t<tk):
        k1=f(vec)
        k2=f(vec+k1*h/2)
        k3=f(vec+k2*h/2)
        k4=f(vec+k3*h)
        vec=vec+(k1+2*k2+2*k3+k4)/6*h
        t=t+h
        sol.append((t,vec))
    return sol
a,b,c,d=2/3,-4/3,-1,0.5
lotka_volterra_solution=lotka_volterra(a,b,c,d,0.7,0.5,0,0.01,30)
def jacobian0(a,b,c,d):
    x=0
    y=0
    return np.array([[a+b*y,b*x],[d*y,c+d*x]])
def jacobian1(a,b,c,d):
    x=-c/d
    y=-a/b
    return np.array([[a+b*y,b*x],[d*y,c+d*x]])
w0,v0 = np.linalg.eig(jacobian0(a,b,c,d))
w1,v1 = np.linalg.eig(jacobian1(a,b,c,d))
print("Problem 7: Lotka-Volterra")
print("at (0,0)")
print("eigenvalues:", w0)
print("eigenvectors:", v0)
print("at", (-a/b,-c/d))
print("eigenvalues:", w1)
print("eigenvectors:", v1)
plt.title("Problem 7: Lotka-Volterra")
plt.plot(*zip(*lotka_volterra_solution))
plt.show()



#DIFFUSION
fig, ax = plt.subplots()

L=10
n=20
grid=np.linspace(0,L,n)
diff_init=np.sin(math.pi*grid/L)
line,=ax.plot(grid,diff_init)
diff_matrix=np.diag(np.ones(n)*-2)+np.diag(np.ones(n-1),-1)+np.diag(np.ones(n-1),+1)
diff_matrix[0,0]=1
diff_matrix[-1,-1]=1
diff_matrix[0,1]=0
diff_matrix[-1,-2]=0
def diffusion(y0,t0,h,tk):
    sol=[y0]
    y=y0
    t=t0
    while(t<tk):
        k1=np.matmul(diff_matrix,y)
        k2=np.matmul(diff_matrix,y+h*k1/2)
        k3=np.matmul(diff_matrix,y+h*k2/2)
        k4=np.matmul(diff_matrix,y+h*k3)
        y=y+(k1+2*k2+ 2*k3+k4)/6*h
        y[0]=0
        y[-1]=0
        sol.append(y)
        t=t+h
    return sol
diff_solution=diffusion(diff_init,0,0.01,100.0)
def animate(i):
    line.set_ydata(diff_solution[i])  # update the data.
    return line,

ani = animation.FuncAnimation(fig, animate,frames=10000, interval=1, blit=True, save_count=10)
plt.title("Problem 8: Diffusion")
plt.show()


