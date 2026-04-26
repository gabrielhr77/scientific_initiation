import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


#-------------------pegando os dados-------------------
data = pd.read_csv('dados/mnist_test.csv')

#print(data.head())

data = np.array(data)
m,n=data.shape #M é as linhas e N é as colunas
np.random.shuffle(data)

data_teste=data[0:1000].T #dados para a comparação posterior
Y_teste=data_teste[0]
X_teste=data_teste[1:n]

data_train=data[1000:m].T #dados para o treinamento da rede
Y_train=data_train[0]
X_train=data_train[1:n]

X_train=X_train/255.0
X_teste=X_teste/255.0

#print(Y_train)


#-------------------funcionamento da rede-------------------
#preciso inicializar os parâmetros

def init_params():
    W1=np.random.rand(10,784)
    #b1=np.random.rand(10,1)
    b1=np.zeros((10,1))
    W2=np.random.rand(10,10)
    #b2=np.random.rand(10,1)
    b2=np.zeros((10,1))
    return W1,b1,W2,b2

def ReLU(Z):# É FUNÇÃO DE ATIVAÇÃO
    return np.maximum(0,Z) #vai em cada elemento de Z (cada Zi), se for maior que 0, retorna Zi, se for menor que 0, retorna Zi

def softmax(Z): #retorna aquela probabilidade definida no NOTES É FUNÇÃO DE ATIVAÇÃO
    #return np.exp(Z)/np.sum(np.exp(Z)) #é a soma das linhas (retorna a soma dos elementos de cada coluna, retornando uma linha só com o valor da soma de sua coluna)
    expZ=np.exp(Z-np.max(Z,axis=0,keepdims=True))
    return expZ/np.sum(expZ,axis=0,keepdims=True)

def forw_prop(W1,b1,W2,b2,X):
    Z1= W1.dot(X)+b1
    A1=ReLU(Z1)
    Z2=W2.dot(A1)+b2
    A2=softmax(Z2)
    return Z1,A1,Z2,A2

def one_hot(Y):
    one_hot_Y=np.zeros((Y.size,Y.max()+1)) #Y.size é M, Y.max()+1 é o número de outputs da rede
    one_hot_Y[np.arange(Y.size),Y]=1
    return one_hot_Y.T

def deriv_ReLU(Z):
    return Z>0 #retorna o booleano (true=1)

def back_prop(Z1,A1,Z2,A2,W2,X,Y):
    m=Y.size
    one_hot_Y=one_hot(Y)
    dZ2=A2 - one_hot_Y
    dW2=1/m*dZ2.dot(A1.T)
    db2=1/m*np.sum(dZ2,axis=1,keepdims=True)
    dZ1=W2.T.dot(dZ2)*deriv_ReLU(Z1)
    dW1=1/m*dZ1.dot(X.T)
    db1=1/m*np.sum(dZ1,axis=1,keepdims=True)
    return dW1,db1,dW2,db2

def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,a):
    W1=W1-a*dW1
    b1=b1-a*db1
    W2=W2-a*dW2
    b2=b2-a*db2
    return W1,b1,W2,b2

def get_predicoes(A):
    return np.argmax(A,0)#ARGMAX retorna o índice de maior valor, ou seja, escolhe a maior probabilidade dentro das probabilidades calculadas pelo SOFTMAX. é uma das etapas de PÓS-PROCESSAMENTO

def get_accuracy(predicao,Y):
    #print(predicao,Y)
    return np.sum(predicao==Y)/Y.size

#aqui é calculado o LOSS
#tem que receber os dados treinados e os dados de teste 
def gradiente_descendente(X_train,Y_train,X_teste,Y_teste,iterac,a): 
    W1,b1,W2,b2=init_params()
    for i in range(iterac):
        Z1,A1,Z2,A2=forw_prop(W1,b1,W2,b2,X_train)
        dW1,db1,dW2,db2=back_prop(Z1,A1,Z2,A2,W2,X_train,Y_train)
        W1,b1,W2,b2=update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,a)
        
        if (i+1)%500==0:
            predicaoTreino=get_predicoes(A2)
            acuraciaTreino=get_accuracy(predicaoTreino,Y_train)

            #preciso fazer uum forward propagation para encontrar as predições de teste
            _,_,_,A2_teste=forw_prop(W1,b1,W2,b2,X_teste)
            predicaoTeste=get_predicoes(A2_teste)
            acuraciaTeste=get_accuracy(predicaoTeste,Y_teste)

            print("\nIteracao numero: ",i+1)
            print(f"Precisao TREINO: {acuraciaTreino}, TESTE: {acuraciaTeste}")
            #print("PRECISAO: ",get_accuracy(get_predicoes(A2),Y_train))
    return W1,b1,W2,b2
    


def faz_predicoes(X,W1,b1,W2,b2):
    _,_,_,A2=forw_prop(W1,b1,W2,b2,X)
    predicoes=get_predicoes(A2)
    return predicoes

def testar_predicoes(index,W1,b1,W2,b2):
    imagemAtual=X_train[:,index,None].reshape(-1,1)
    predicao=faz_predicoes(imagemAtual,W1,b1,W2,b2)
    label=Y_train[index]
    print("PREDICAO: ",predicao)
    print("LABEL: ",label)
    imagemAtual=imagemAtual.reshape((28,28))*255
    '''plt.gray()
    plt.imshow(imagemAtual,interpolation='nearest')
    plt.show()'''


#-------------------instância da rede-------------------

W1,b1,W2,b2=gradiente_descendente(X_train,Y_train,X_teste,Y_teste,3000,0.01)
for i in range(10):    
    testar_predicoes(i,W1,b1,W2,b2)