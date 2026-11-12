import math as mt, numpy as np, pylab as pl

def f(x,t):
    return -x**3 + mt.sin(t)

a=0.0
b=10.0
N=1000
h=(b-a)/N
print(b-a)
print((b-a)/N)
x=0.0

tpoints=np.arange(a,b,h)
xpoints=[]
for t in tpoints:
    xpoints.append(x)
    k1=h*f(x,t)
    k2=h*f(x+0.5*k1,t+0.5*h)
    x+=k2

pl.plot(tpoints,xpoints)
pl.xlabel("t")
pl.ylabel("x(t)")
pl.show()
