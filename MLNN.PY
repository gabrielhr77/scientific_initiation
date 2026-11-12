import numpy as np
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt


#criando os dados
N=300                       #quantidade de dadis criados
x=t.randn(N,1)              #geração dos X  
y=x+t.randn(N,1)/2          #geração dos Y


ANNregressao=nn.Sequential(
    nn.Linear(1,128),
    nn.Tanh(),
    nn.Linear(128,128),
    nn.Tanh(),
    nn.Linear(128,1)  # 3 corpos em 2D
)

ANNregressao                #cria a instância da rede neural definida acima




learningRate=.05            #taxa de aprendizado (o quanto os dados são alterados a cada iteração para que a rede possa ter suas mutações e encontrar a versão mais adequada para descrever o fenômeno sendo estudado)
lossFun=nn.MSELoss()        #função de loss (como usarei PINNs essa função será alterada)   
#aplicando o gradiente descendente
optimizer=t.optim.SGD(ANNregressao.parameters(),lr=learningRate)

#treinando o modelo
epocas=500
losses=t.zeros(epocas)      #gera um vetor do tamanho da quantidade de iterações que serão feitas para treinar a rede


for epoca in range(epocas):
    #forward pass
    output=ANNregressao(x)

    #computando a loss
    loss=lossFun(output,y)
    losses[epoca]=loss

    #backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#computando manualmente as losses
#forward pass final
predicao=ANNregressao(x)

#calculo da MSE
testloss=(predicao-y).pow(2).mean()

plt.plot(losses.detach(),'o',markerfacecolor='w',linewidth=.1)
plt.plot(epocas,testloss.detach(),'ro')
plt.xlabel('Epocas')
plt.ylabel('Loss')
plt.show()

plt.plot(x,y,'bo',label='Dados reais')
plt.plot(x,predicao.detach(),'rs',label='Predicao')
plt.legend()
plt.show()