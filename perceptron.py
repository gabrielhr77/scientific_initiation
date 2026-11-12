import numpy as np
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
#from IPython import display
#display.set_matplotlib_formats('svg')

#criando os dados
N=300
x=t.randn(N,1)
y=x+t.randn(N,1)/2

#plotar
#plt.plot(x,y,'s')
#plt.show()
'''
#criando o modelo de rede neural com 1 neuronio de entrada e 1 oculto
ANNregressao=nn.Sequential(
    nn.Linear(1,1),#input layer
    nn.ReLU(),     #funcao de ativacao
    #nn.Sigmoid(),
    nn.Linear(1,1) #output layer
)
ANNregressao'''
#criando o modelo de rede neural com 2 neuronios de entrada (com a mesma entrada), 4 oculto e 1 de saida
ANNregressao=nn.Sequential(
    nn.Linear(1,2),#input layer
    nn.ReLU(),     #funcao de ativacao
    #nn.Sigmoid(),
    nn.Linear(2,4), #4 ocultos
    nn.ReLU(),
    #nn.Sigmoid(),
    nn.Linear(4,1)#neuronio de saida
)
ANNregressao

'''
#dados do modelo
print(ANNregressao[0].weight)
print(ANNregressao[0].bias)

print("\nPeso da primeira camada:")
print(ANNregressao[0].weight)

print("\nBias da primeira camada:")
print(ANNregressao[0].bias)

print("\nShape do peso:", ANNregressao[0].weight.shape)'''


learningRate=.05
lossFun=nn.MSELoss()  #loss function
#aplicando o gradiente descendente
optimizer=t.optim.SGD(ANNregressao.parameters(),lr=learningRate)

#treinando o modelo
epocas=500
losses=t.zeros(epocas)


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

#computando manialmente as losses
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