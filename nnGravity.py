import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch as tor
import torch.nn as nn
from torch.autograd import grad


def importEstadoNPZ(a):
    dados=np.load(f"dados/simulacao{a}.npz")
    return dados['trajetoria'],dados['massas']

trajetoria,massas=importEstadoNPZ(1)

print(f"M1 {massas[0]}, M2 {massas[1]}, M3 {massas[2]}")

x1=trajetoria[:,0]
y1=trajetoria[:,1]
x2=trajetoria[:,4]
y2=trajetoria[:,5]
x3=trajetoria[:,8]
y3=trajetoria[:,9]

plt.scatter(x1[0], y1[0], color='green', s=100, label='Início', zorder=5)
plt.scatter(x1[-1], y1[-1], color='red', s=100, label='Fim', zorder=5)
plt.scatter(x2[0], y2[0], color='green', s=100, label='Início', zorder=5)
plt.scatter(x2[-1], y2[-1], color='red', s=100, label='Fim', zorder=5)
plt.scatter(x3[0], y3[0], color='green', s=100, label='Início', zorder=5)
plt.scatter(x3[-1], y3[-1], color='red', s=100, label='Fim', zorder=5)

plt.plot(x1, y1, 'b-', label='Real 1', alpha=0.5, linewidth=3)
plt.plot(x2, y2, 'g-', label='Real 2', alpha=0.5, linewidth=3)
plt.plot(x3, y3, 'c-', label='Real 3', alpha=0.5, linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Todas as trajetórias')
#plt.legend()
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()

class TresCosposPINN(nn.Module):
    def __init__(self):
        super(TresCosposPINN,self).__init__() #chama o construtor da classe pai (Module)
        self.net=nn.Sequential(
            nn.Linear(1,8),
            nn.Tanh(),
            nn.Linear(8,8),
            nn.Tanh(),
            nn.Linear(8,6)  # 3 corpos em 2D
        )
    def forward(self,tempo):
        return self.net(tempo)

