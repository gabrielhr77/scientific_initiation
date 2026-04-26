import torch as tc
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import numpy as np



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
            nn.Linear(2,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,2)
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

    def forward(self,x):
        return self.net(x)
        

modelo=RedeGravitacional()
loss_padrao=nn.MSELoss()
optimizer=tc.optim.Adam(modelo.parameters(),lr=1e-3)

#preparando o dataset
X_tensor=tc.tensor(X,dtype=tc.float32)
Y_tensor=tc.tensor(Y,dtype=tc.float32)

dataSet=TensorDataset(X_tensor,Y_tensor)

trainLoader=DataLoader(dataSet,batch_size=64,shuffle=True)
print('..')
for inputs, targets in trainLoader:
    print(inputs.shape)
    print(targets.shape)
    break

#rodando o loop de treino
for epoca in range(10):
    print(f'\nComecando epoca {epoca+1}\n')
    loss_epoca=0.0
    for inputs, alvos in trainLoader:
        predicao=modelo(inputs)
        loss=loss_padrao(predicao,alvos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoca+=loss.item()
    loss_medio=loss_epoca/len(trainLoader)
    print(f'Epoca {epoca+1}: {loss_medio}\n')

teste=tc.tensor([[3/5,4/5]],dtype=tc.float32)
print(f'TESTE COM VALOR DESCONHECIDO: {modelo(teste)}') #predição normalizada da rede [ax_pred,ay_pred]
pred = modelo(teste).detach().numpy()

pred_real = pred * sigma

print(pred_real[0]) #valores de [ax_pred,ay_pred] DESNORMALIZADOS