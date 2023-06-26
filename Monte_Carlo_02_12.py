#Jerzy Knopik 03.12.2022

import math
import numpy as np
from scipy.integrate import *
import matplotlib.pyplot as plt


# Problem 1
def pifunction(Nsamples):
    x=np.random.uniform(0,1,[2,Nsamples])
    r2=np.sum(x*x,axis=0)
    return len(r2[r2<=1])/Nsamples
xdata=[]
ydata=[]
for i in range(1000,100000,1000):
    xdata.append(i)
    ydata.append(abs(4*pifunction(i)-math.pi))
plt.figure(figsize=(10,5))
#plt.yscale('log')
plt.title("Problem 1: Error")
plt.plot(xdata,ydata)
plt.show()

#Problem 3

def mcballvolume(dim, Nsamples):
    x=np.random.uniform(0,1,[dim,Nsamples])
    r2=np.sum(x*x,axis=0)
    return len(r2[r2<=1])/Nsamples*(2**dim)
def ballvolume(dim):
    return (math.pi)**(dim/2)/math.gamma(dim/2+1)

def quadballvolume(dim):
    result=quad(lambda z: dim*math.exp(-z*z)*z**(dim-1),0,np.inf,full_output=1)
    integral=result[0]
    volume=(math.pi)**(dim/2)/integral
    print("Volume  : ", volume)
    print("error :", volume-ballvolume(dim))
    print("number of evaluations: ", result[2]['neval'])
print("MonteCarlo")    
print("error :",mcballvolume(8,1000000)-ballvolume(8))
print("Quad")
quadballvolume(8)

#Problem 4

x1=np.random.uniform(0,1,1000000)
x2=np.random.uniform(0,1,1000000)

variance= 4
def normal(x):
    return np.exp(-0.5*(x/(variance))**2)/((2*math.pi)**(1/2)*variance)

y1=(variance)*(-2*np.log(x1))**(1/2)*np.cos(2*math.pi*x2)
y2=(-2*np.log(x1))**(1/2)*np.sin(2*math.pi*x2)

xdata=np.linspace(-15,15,200)
plt.hist(y1, bins=np.linspace(-15, 15, 100), density='True', label='generated dist.')
plt.plot(xdata,normal(xdata))
plt.title("Box-Muller method")
plt.show()

#Problem 7
tr_sample=10000
time_sample=240
time=np.arange(0,time_sample+1,1)
x=np.ones((tr_sample,time_sample+1))
W=np.random.normal(0.00,1.,[tr_sample,time_sample])
alpha=1e-3
for i in range(time_sample):
    x[:,i+1]=(1+alpha*W[:,i])*x[:,i]
for i in range(10):
    plt.plot(time,x[i,:])
plt.show()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.hist(x[:,i*40+30], bins=np.linspace(0.95, 1.05, 100), density='True', label='generated dist.',color='C'+str(2*i))
plt.show()
