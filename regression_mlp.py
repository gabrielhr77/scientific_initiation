import torch as tc
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
import matplotlib.pyplot as plt


G=1
M=1
alpha=1e-8

n=50000

X=[]
Y=[]
#gera pontos [x,y] aleatórios no espaço e calcula a aceleração gravitacional gerada levando em consideração uma massa atraindo esses pontos no centro do plano cartesiano
for _ in range(n):
    while True:
        x=np.random.uniform(-5,5)
        y=np.random.uniform(-5,5) #aqui tive o problema de que o dataset estava gerando dados muito proximos ao ZERO, então o dataset estava cheio de OUTLIERS
        
        r=np.sqrt(x**2+y**2)+alpha
        #softening físico evitando cálculos extremos - se for menor que 1.5 ele continua no loop do while para refazer o ponto
        if r>1.5:
            break

    ax=-G*M*x/r**3
    ay=-G*M*y/r**3

    X.append([x,y])
    Y.append([ax,ay])

X=np.array(X)

X=X/5.0 #normalizando
Y=np.array(Y)
#Y=Y/np.max(np.abs(Y))
sigma=np.std(Y,axis=0)
Y=Y/np.std(Y,axis=0) #normalizando
print('.')
print(np.max(np.abs(Y)))
print(np.mean(np.abs(Y)))


class RedeGravitacional(nn.Module):
    #esse é o construtor
    def __init__(self):
        #inicia a superclasse
        super().__init__()

        '''self.net=nn.Sequential(
            nn.Linear(2,8),
            nn.Tanh(),
            nn.Linear(8,8),
            nn.Tanh(),
            nn.Linear(8,2)
        )'''
        '''self.net=nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,2)
        )'''
        self.net=nn.Sequential(
            nn.Linear(2,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,2)
        )
        '''self.net=nn.Sequential(
            nn.Linear(2,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,2)
        )'''

    def forward(self,x):
        return self.net(x)
        

modelo=RedeGravitacional()
loss_padrao=nn.MSELoss()
optimizer=tc.optim.Adam(modelo.parameters(),lr=1e-3)

#preparando o dataset
X_tensor=tc.tensor(X,dtype=tc.float32)
Y_tensor=tc.tensor(Y,dtype=tc.float32)

dataSet=TensorDataset(X_tensor,Y_tensor)

#trainLoader=DataLoader(dataSet,batch_size=128,shuffle=True)
trainLoader=DataLoader(dataSet,batch_size=64,shuffle=True)
print('..')
for inputs, targets in trainLoader:
    print(inputs.shape)
    print(targets.shape)
    break
historico_loss=[]
#rodando o loop de treino
for epoca in range(10):
    print('\n-----------------------------------------------')
    print(f'Comecando epoca {epoca+1}')
    loss_epoca=0.0

    for inputs, alvos in trainLoader:
        predicao=modelo(inputs)
        loss=loss_padrao(predicao,alvos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoca+=loss.item()
    loss_medio=loss_epoca/len(trainLoader)
    historico_loss.append(loss_medio)
    print(f'Epoca {epoca+1}: {loss_medio}')

teste=tc.tensor([[3/5,4/5]],dtype=tc.float32)
print(f'TESTE COM VALOR DESCONHECIDO: {modelo(teste)}') #predição normalizada da rede [ax_pred,ay_pred]
pred = modelo(teste).detach().numpy()

pred_real = pred * sigma

print(pred_real[0]) #valores de [ax_pred,ay_pred] DESNORMALIZADOS


#PLOT GERADO POR IA PARA OBSERVAR APRENDIZADO PELA REDE (não autoral)
# grade de pontos 
xs=np.linspace(-5,5,20) 
ys=np.linspace(-5,5,20) 

Xg,Yg=np.meshgrid(xs,ys) 

Ux=np.zeros_like(Xg) 
Uy=np.zeros_like(Yg) 
for i in range(len(xs)): 
    for j in range(len(ys)): 
        x=Xg[j,i] 
        y=Yg[j,i] 
        r=np.sqrt(x*x+y*y) 
        # evita singularidade central 
        if r<1.5: 
            continue # mesma normalização usada no treino 
        entrada=tc.tensor( [[x/5,y/5]], dtype=tc.float32 ) 
        pred=modelo(entrada).detach().numpy()[0] # desnormaliza 
        pred=pred*sigma 
        
        Ux[j,i]=pred[0] 
        Uy[j,i]=pred[1] 
mag=np.sqrt(Ux**2+Uy**2) 
#vetores normalizados
#Ux_plot=Ux/(mag+1e-8)
#Uy_plot=Uy/(mag+1e-8)

fig,ax=plt.subplots(1,2,figsize=(14,6))
ax[0].plot(historico_loss)
ax[0].set_yscale('log')
ax[0].set_title("Curva de aprendizado")
ax[0].set_xlabel("Épocas")
ax[0].set_ylabel("Loss (MSE)")

#q=ax[1].quiver(Xg,Yg,Ux_plot,Uy_plot,mag) 
q=ax[1].quiver(Xg,Yg,Ux,Uy,mag) 
fig.colorbar(q,ax=ax[1],label="Magnitude da aceleração")

ax[1].set_title("Campo gravitacional aprendido pela rede")

ax[1].axis('equal')

plt.tight_layout() 
plt.show()