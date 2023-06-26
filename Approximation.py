import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd

# We define functions
def f1(x):
    return 1/(x*x+1)

def f2(x):
    return np.exp(-x*x)

L=10 #size of the domain

grid=np.linspace(-L/2,L/2,1000)

def lincoll(n):
    return np.linspace(-L/2,L/2,n)

def l(x,k,coll): #lagrange polynomials
    result=1
    for i in range(0, len(coll)):
        if (i!=k):
            result = result*(x-coll[i])/(coll[k]-coll[i])
    return result

def approximation(x,f,coll):
    result=0
    for i in range(0,len(coll)):
        result+=f(coll[i])*l(x,i,coll)
    return result


error1=[]
error2=[]
for i in range(1,65):
    error1.append(np.sum(np.abs(f1(grid)-approximation(grid,f1,lincoll(i)))))    
    error2.append(np.sum(np.abs(f2(grid)-approximation(grid,f2,lincoll(i)))))

error1_dict=dict(zip(range(1,65),error1))
error2_dict=dict(zip(range(1,65),error2))

error1_number=min(error1_dict, key=error1_dict.get)
error2_number=min(error2_dict, key=error2_dict.get)


plt.plot(grid,approximation(grid,f1,lincoll(error1_number)),'r')
plt.plot(grid,f1(grid),'b')
plt.title('1/(x*x+1) approx. by Lagrange polynomials \n with equidistant grid \n minimal error: %0.2g for n=%d' % (error1[error1_number], error1_number))
plt.show()

# We cannot approximate f1 correctly due to Runge phenomenon

plt.plot(grid,approximation(grid,f2,lincoll(error2_number)),'r')
plt.plot(grid,f2(grid),'b')
plt.title('exp(-x*x) approx. by Lagrange polynomials \n with equidistant grid \n minimal erroror: %0.2g for n=%d' % (error2[error2_number], error2_number))
plt.show()                      


def coscoll(n):    #Chebyshev collocation points
    return 5*np.cos(np.linspace(0,np.pi,n))

coserror1=[]
coserror2=[]

for i in range(70,120):
    coserror1.append(np.sum(np.abs(f1(grid)-approximation(grid,f1,coscoll(i)))))    
    coserror2.append(np.sum(np.abs(f2(grid)-approximation(grid,f2,coscoll(i)))))

coserror1_dict=dict(zip(range(70,120),coserror1))
coserror2_dict=dict(zip(range(70,120),coserror2))

coserror1_number=min(coserror1_dict, key=coserror1_dict.get)
coserror2_number=min(coserror2_dict, key=coserror2_dict.get)

plt.plot(grid,approximation(grid,f1,coscoll(coserror1_number)),'r')
plt.plot(grid,f1(grid),'b')
plt.title('1/(x*x+1) approx. by Lagrange polynomials \n  with Chebyshev grid \n minimal error: %0.2g for n=%d' % (coserror1[coserror1_number-70],coserror1_number))
plt.show()

plt.plot(grid,approximation(grid,f2,coscoll(coserror2_number)),'r')
plt.plot(grid,f2(grid),'b')
plt.title('exp(-x*x) approx. by Lagrange polynomials \n with Chebyshev grid \n minimal error: %0.2g for n=%d' % (coserror2[coserror2_number-70],coserror2_number))
plt.show()

def csf1(n):
    return scipy.interpolate.CubicSpline(lincoll(n), f1(lincoll(n)), axis=0, bc_type='not-a-knot', extrapolate=None)
def csf2(n):
    return scipy.interpolate.CubicSpline(lincoll(n), f2(lincoll(n)), axis=0, bc_type='not-a-knot', extrapolate=None)

cserror1=[]
cserror2=[]

for i in range(50,150):
    cserror1.append(np.sum(np.abs(f1(grid)-csf1(i).__call__(grid))))   
    cserror2.append(np.sum(np.abs(f2(grid)-csf2(i).__call__(grid))))

plt.yscale('log')
plt.plot(range(50,150),cserror1,'r')
plt.plot(range(50,150),cserror2,'b')
plt.title('Error max|f-f_approx| in a logarithmic scale for a CubicSpline approximation \n red - 1/(x*x+1) \n blue - exp(-x*x) ')
plt.show()

def sincgrid(l):
    return np.linspace(-l/2,l/2,2000)

def sinccoll(n,l):
    return np.linspace(-l/2,l/2,n)
    
def sincapproximation(x,f,coll,l):
    result=0
    for i in range(0,len(coll)):
        result+=f(coll[i])*np.sinc((x-coll[i])*(len(coll)-1)/l)
    return result

sincerror1=[]
sincerror2=[]
sincerror3=[]

sincgrid1=sincgrid(2*L)
sincgrid2=sincgrid(4*L)
sincgrid3=sincgrid(8*L)
g=f2
for i in range(20,300):
    sincerror1.append(np.sum(np.abs(g(sincgrid1)-sincapproximation(sincgrid1,g,sinccoll(i,2*L),2*L))))
    sincerror2.append(np.sum(np.abs(g(sincgrid2)-sincapproximation(sincgrid2,g,sinccoll(i,4*L),4*L))))
    sincerror3.append(np.sum(np.abs(g(sincgrid3)-sincapproximation(sincgrid3,g,sinccoll(i,8*L),8*L))))


plt.yscale('log')
plt.plot(range(20,300),sincerror1,'r', label ='2L')
plt.plot(range(20,300),sincerror2,'g', label ='4L')
plt.plot(range(20,300),sincerror3,'b', label ='8L')
plt.legend()
plt.title('Error max|f-f_approx| in a logarithmic scale for a sinc approximation \n for f= exp(-x*x) and three different values of L')
plt.show()

#sincplot(f2,5*L,80)

# For approximation with sinc: when we increase L we need to increase a number
# of collocation points too, but eventually the approximation converges to the
# solution for both f1 and f2.
