import math
import numpy as np
def golden_ratio_minimization(f, a, b, prec):
    k = (math.sqrt(5)-1)/2
    c = a+(b-a)*k
    d = b-(b-a)*k
    while (b-a)>prec:
        if (f(c)<f(d)):
            a = d
            d = c
            c = a+(b-a)*k
        else:
            b = c
            c = d
            d = b-(b-a)*k
    return (a+b)/2

g = lambda x : x*x-3*x+1
h = lambda x : (4*x+math.exp(x))/(2*math.exp(x)+3*math.exp(-2*x)+7*x*x)
print("golden ratio",golden_ratio_minimization(g,1,6,1e-15))
print("golden ratio",golden_ratio_minimization(h,-1,0,1e-15))
print("golden ratio",golden_ratio_minimization(h,2,5,1e-15))
print("golden ratio, eror for sine func",math.pi*3/2-golden_ratio_minimization(math.sin,4,6,1e-15))

#To jest wersja algorytmu, którą Brent opublikował w swojej książce
#"Algorithms for minimization without derivatives".Prawdę mówiąc
# bardzo trudno było mi znaleźć w internecie inną wersję niż ta podana
# przez Brenta. Książka "Numerical receips in C. The art of scientific computing"
#przepisuje algorytm Brenta kropka w kropkę. Jedynym miejscem, w którym  znalazłem
# uproszczony algorytm jest film na youtube: https://www.youtube.com/watch?v=9Zejl2YzaYY

def brent_minimization(f, a, b, prec):
    k=(math.sqrt(5)-1)/2
    v=w=x=b-k*(b-a)
    fx=fw=fv=f(x)
    e=0
    while(max(b-x,x-a)>2*prec):
        m=0.5*(a+b)
        p=q=r=0
        if abs(e)>prec: #jeśli przedostatni krok był większy od założonego błędu
            r=(x-w)*(fx-fv)
            q=(x-v)*(fx-fw)
            p=-np.sign(q-r)*((x-v)*q-(x-w)*r)
            q=2*abs(q-r)
            r=e
            e=d #aktualizujemy przedostatni krak
        if abs(p)<0.5*abs(q*r) and p>q*(a-x) and p<q*(b-x): #jeśli nowy krok jest mniejszy o pół od przedostatniego
            #print("parabolic")                             # i wierzchołek nie jest poza przedziałem
            d=p/q                                           # przy okazji sprawdza że q nie jest równe zero
            u=x+d
            if min(u-a,b-u)<2*prec:    # jeśli nowy punkt jest za blisko brzegu to go odsuwamy o prec
                d=np.sign(m-x)*prec 
        else:
            #print("golden ratio")
            if x>=m:
                e=(a-x)
                d=(1-k)*e
            else:
                e=(b-x)
                d=(1-k)*e
        u=x+np.sign(d)*max(abs(d),prec)  #jak zmiana jest mniejsza od prec to zaokrąglamy do prec
        fu=f(u)
        if fu<=fx:
            if u>=x:
                a=x
            else:
                b=x
            v,w,x=w,x,u
            fv,fw,fx=fw, fx, fu
        else:
            if u<x:
                a=u
            else:
                b=u
            if fu <=fw or w==x:
                v,w=w,u
                fv,fw=fw,fu
            elif fu<=fv or v==x or v==w:
                v=u
                fv=fu
    return x
print("brent minimization", brent_minimization(g,0,10,1e-15))
print("brent minimization", brent_minimization(h,-2,0,1e-15))
print("brent minimization", brent_minimization(h,0,4,1e-15))
print("brent minimization, error for sine func", math.pi*3/2-brent_minimization(math.sin,4,6,1e-15))
# Dla błędu rzędu 1e-8 do 1e-10 algorytm Brenta znajduje minimum w kilku iteracjach,
# kończąc iteracjami parabolicznymi. Dla większej precyzji, rzędu 1e-12 algorytm kończy
#szukać minimum za pomocą złotego podziału. Nie wiem dlaczego tak jest.
