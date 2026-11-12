'''import numpy as np, matplotlib.pyplot as plt





def fx(x):
    return 3*x**2 -3*x +4

def dfx(x):
    return 6*x-3

#definindo uma range para o x de -2 a 2 com 2001 passos/steps
x=np.linspace(-2,2,2001)

localmin=np.random.choice(x,1)

taxaAprendizado=.01
ciclos=315

for i in range(ciclos):
    grad=dfx(localmin)
    localmin=localmin-taxaAprendizado*grad
    print(localmin)

plt.plot(x,fx(x),x,dfx(x))
plt.plot(localmin,dfx(localmin),'ro')
plt.plot(localmin,fx(localmin),'ro')
plt.xlim(x[[0,-1]])
plt.grid()
plt.show()'''





'''
import numpy as np, matplotlib.pyplot as plt

PI=3.14159265359


def fx(x):
    return np.cos(2*np.pi*x)+x**2

def dfx(x):
    return 2*PI*np.sin(2*np.pi*x)+2*x

#definindo uma range para o x de -2 a 2 com 2001 passos/steps
x=np.linspace(-2,2,2001)

localmin=np.random.choice(x,1)

taxaAprendizado=.01
ciclos=315

for i in range(ciclos):
    grad=dfx(localmin)
    localmin=localmin-taxaAprendizado*grad
    print(localmin)

plt.plot(x,fx(x),x,dfx(x))
plt.plot(localmin,dfx(localmin),'ro')
plt.plot(localmin,fx(localmin),'ro')
plt.xlim(x[[0,-1]])
plt.grid()
plt.show()'''





import numpy as np, matplotlib.pyplot as plt,sympy as sym

PI=3.14159265359


def peaks(x,y):
    x,y=np.meshgrid(x,y)
    z=3*(1-x)**2*np.e**(-x**2-(y+1)**2)-10*(x/5-x**3-y**5)*np.e**(-x**2-y**2)-(np.e**(-(x+1)**2-y**2))/3
    return z

def dfx(x):
    return 2*PI*np.sin(2*np.pi*x)+2*x

#definindo uma range para o x de -2 a 2 com 2001 passos/steps
x=np.linspace(-3,3,201)
y=np.linspace(-3,3,201)

Z=peaks(x,y)

plt.imshow(Z,extent=[x[0],x[-1],y[0],y[-1]],vmin=-5,vmax=5,origin='lower')
#plt.show()

sx,sy=sym.symbols('sx,sy')
sZ=3*(1-sx)**2*sym.exp(-sx**2-(sy+1)**2)-10*(sx/5-sx**3-sy**5)*sym.exp(-sx**2-sy**2)-(sym.exp(-(sx+1)**2-sy**2))/3

df_x=sym.lambdify((sx,sy),sym.diff(sZ,sx),'sympy')
df_y=sym.lambdify((sx,sy),sym.diff(sZ,sy),'sympy')

print(df_x(1,1).evalf())

localmin=np.random.rand(2)*4-2
localmax=localmin
ponto_inicial1=localmin[:]
ponto_inicial2=localmax[:]

taxaAprendizado=.01
ciclos1=10000
ciclos2=10000

trajetoriaMin=np.zeros((ciclos1,2))
trajetoriaMax=np.zeros((ciclos2,2))
for i in range(10000):
    grad1=np.array([df_x(localmin[0],localmin[1]).evalf(),df_y(localmin[0],localmin[1]).evalf()])
    grad2=np.array([df_x(localmax[0],localmax[1]).evalf(),df_y(localmax[0],localmax[1]).evalf()])
    localmin=localmin-taxaAprendizado*grad1
    localmax=localmax+taxaAprendizado*grad2
    trajetoriaMin[i,:]=localmin 
    trajetoriaMax[i,:]=localmax

'''print(localmin)
print(ponto_inicial1)
print(ponto_inicial2)'''

plt.imshow(Z,extent=[x[0],x[-1],y[0],y[-1]],vmin=-5,vmax=5,origin='lower')
plt.plot(ponto_inicial1[0],ponto_inicial1[1],'bs')
plt.plot(ponto_inicial2[0],ponto_inicial2[1],'bs')
plt.plot(localmin[0],localmin[1],'ro')
plt.plot(trajetoriaMin[:,0],trajetoriaMin[:,1],'r')
plt.plot(localmax[0],localmax[1],'ro')
plt.plot(trajetoriaMax[:,0],trajetoriaMax[:,1],'r')
plt.colorbar()
plt.show()
'''
localmin=np.random.choice(x,1)

taxaAprendizado=.01
ciclos=315

for i in range(ciclos):
    grad=dfx(localmin)
    localmin=localmin-taxaAprendizado*grad
    print(localmin)

plt.plot(x,fx(x),x,dfx(x))
plt.plot(localmin,dfx(localmin),'ro')
plt.plot(localmin,fx(localmin),'ro')
plt.xlim(x[[0,-1]])
plt.grid()
plt.show()'''