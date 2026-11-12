import numpy as np
import torch as t
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
import os




def carregarEstadosNPZ(nome):
    dados=np.load(f"simulacoesArtificiais/simulacao{nome}.npz")
    return dados['trajetoria'], dados['tempo'], dados['massas']

#importando os dados de um arquivo NPZ ("C:\Users\gabri\OneDrive\Desktop\USP\IC\programas\simulacoesArtificiais\simulacao0.npz")

dados,tempo,massas=carregarEstadosNPZ(1)

print(dados)





m1=massas[0]
m2=massas[1]
m3=massas[2]

x1=dados[:,0]
y1=dados[:,1]

x2=dados[:,4]
y2=dados[:,5]

x3=dados[:,8]
y3=dados[:,9]

tempo=dados[:,10]

nTotal=len(tempo)/5 #pega 20% do total de passos
nAmostras=1000

indices=np.linspace(0,nTotal-1,nAmostras,dtype=int) #isso gera uma amostragem uniforme

tempo=tempo[indices]
x1=x1[indices]
x2=x2[indices]
x3=x3[indices]
y1=y1[indices]
y2=y2[indices]
y3=y3[indices]

if len(tempo.shape)==1:
    tempo=tempo.reshape(-1,1)

#gerando o vetor que será passado ao modelo
tx1 = t.tensor(x1, dtype=t.float32).reshape(-1, 1)
ty1 = t.tensor(y1, dtype=t.float32).reshape(-1, 1)
tx2 = t.tensor(x2, dtype=t.float32).reshape(-1, 1)
ty2 = t.tensor(y2, dtype=t.float32).reshape(-1, 1)
tx3 = t.tensor(x3, dtype=t.float32).reshape(-1, 1)
ty3 = t.tensor(y3, dtype=t.float32).reshape(-1, 1)
dadosCompletos=[]
dadosCompletos=t.tensor(dadosCompletos)
dadosCompletos = t.cat((tx1, ty1, tx2, ty2, tx3, ty3), dim=1)

#plot do NPZ
plt.plot(x1, y1, label='corpo 1')
plt.plot(x2, y2, label='corpo 2')
plt.plot(x3, y3, label='corpo 3')
plt.scatter(x1[0], y1[0])
plt.scatter(x2[0], y2[0])
plt.scatter(x3[0], y3[0])
plt.title("Trajetória do arquivo NPZ")
plt.axis("equal")
plt.grid()
plt.show()

#gerando os TENSORES, uma vez que o torch não trabalha com NDARRAY
# o .reshape(-1, 1) é apenas para arrumar o shaoe do tensor que estou passando da simulação gerada, que possui X passos, logo, (1,X), e não (X,1), que é o que preciso
#para evitar que escalas diferentes no tempo afetem o treinamento da rede eu vou NORMALIZAR o tempo
tempoNormalizado=((tempo-tempo.min())/(tempo.max()-tempo.min())).reshape(-1,1)

tempoTensor=t.tensor(tempoNormalizado,dtype=t.float32, requires_grad=True)
x1=t.tensor(x1, dtype=t.float32).reshape(-1, 1)
y1=t.tensor(y1, dtype=t.float32).reshape(-1, 1)

x2=t.tensor(x2, dtype=t.float32).reshape(-1, 1)
y2=t.tensor(y2, dtype=t.float32).reshape(-1, 1)

x3=t.tensor(x3, dtype=t.float32).reshape(-1, 1)
y3=t.tensor(y3, dtype=t.float32).reshape(-1, 1)


#NORMALIZANDO AS COORDENADAS PARA QUE O TREINAMENTO OCORRA SEM EXTRAPOLAÇÕES DA REDE
coordsMax=max(x1.max(),y1.max(),x2.max(),y2.max(),x3.max(),y3.max())
x1=x1/coordsMax
y1=y1/coordsMax
x2=x2/coordsMax
y2=y2/coordsMax
x3=x3/coordsMax
y3=y3/coordsMax



print("\nCOORDSMAX: ",coordsMax,"\nMINIMO DOS VALORES DE x1,y1,...,x3,y3 APOS NORMALIZAR COM coordsMax: ",min(x1.min(),x1.min(),x2.min(),y2.min(),x3.min(),y3.min()))

#criando uma CLASSE para a rede neural com suas funções básicas (init e forward)
class TresCosposPINN(nn.Module):
    def __init__(self):
        super(TresCosposPINN,self).__init__() #chama o construtor da classe pai (Module)
        '''self.net=nn.Sequential(
            nn.Linear(1,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,6)  # 3 corpos em 2D
        )'''
        '''self.net=nn.Sequential(
            nn.Linear(10,512),
            nn.SiLU(),
            nn.Linear(512,512),
            nn.SiLU(),
            nn.Linear(512,512),
            nn.SiLU(),
            nn.Linear(512,512),
            nn.SiLU(),
            nn.Linear(512,6)  # 3 corpos em 2D
        )'''
        self.net=nn.Sequential(
            nn.Linear(1,8),
            nn.ReLU(),
            nn.Linear(8,8),
            nn.ReLU(),
            nn.Linear(8,6)  # 3 corpos em 2D
        )
    def forward(self,tempo):
        return self.net(tempo)
    
#função para calcular as derivadas
def derivada(x,tempo):
    # X:tensor que queremos derivar
    # t: tensor em relação ao qual queremos derivar
    # grad_outputs=t.ones_like(x): cria um tensor de 1 com o shape de X
    # create_graph=True permite cálculo de derivadas de segunda ordem, mantendo o grafo
    #como grad() retorna uma tupla, só queremos pegar a derivada, o [0]
    return grad(x,tempo,grad_outputs=t.ones_like(x),create_graph=True)[0]

def lossPINN(modelo,t,x1,y1,x2,y2,x3,y3,G,m1,m2,m3,tMin,tMax): #mede o quanto a rede está violando as leis da física
    #FORWARD PASS: obtenho as 6 coordeadas
    outputRede=modelo(t)
    #print(dados)
    #saídas geradas
    x1Pred=outputRede[:,0:1]
    y1Pred=outputRede[:,1:2]
    x2Pred=outputRede[:,2:3]
    y2Pred=outputRede[:,3:4]
    x3Pred=outputRede[:,4:5]
    y3Pred=outputRede[:,5:6]
    

    #derivadas primeira (tempo normalizado)
    vx1n=derivada(x1Pred,t)
    vy1n=derivada(y1Pred,t)
    vx2n=derivada(x2Pred,t)
    vy2n=derivada(y2Pred,t)
    vx3n=derivada(x3Pred,t)
    vy3n=derivada(y3Pred,t)

    #ajustando para o tempo real
    escala=1.0/(tMax-tMin) #fator de escala de tempo para retirar a normalização
    vx1=vx1n*escala
    vy1=vy1n*escala
    vx2=vx2n*escala
    vy2=vy2n*escala
    vx3=vx3n*escala
    vy3=vy3n*escala

    #derivadas segunda (tempo normalizado)
    ax1n=derivada(vx1n,t)
    ay1n=derivada(vy1n,t)
    ax2n=derivada(vx2n,t)
    ay2n=derivada(vy2n,t)
    ax3n=derivada(vx3n,t)
    ay3n=derivada(vy3n,t)

    #ajustando para o tempo real
    ax1=(ax1n*escala**2)*coordsMax
    ay1=(ay1n*escala**2)*coordsMax
    ax2=(ax2n*escala**2)*coordsMax
    ay2=(ay2n*escala**2)*coordsMax
    ax3=(ax3n*escala**2)*coordsMax
    ay3=(ay3n*escala**2)*coordsMax

    #calculando as distâncias
    r12=((x2Pred-x1Pred)*coordsMax)**2 + ((y2Pred-y1Pred)*coordsMax)**2 + epsilon
    r12=r12.sqrt()
    r13=((x3Pred-x1Pred)*coordsMax)**2 + ((y3Pred-y1Pred)*coordsMax)**2 + epsilon
    r13=r13.sqrt()
    r23=((x3Pred-x2Pred)*coordsMax)**2 + ((y3Pred-y2Pred)*coordsMax)**2 + epsilon
    r23=r23.sqrt()

    #calculando as forças nos corpos
    #corpo 1
    f12x=G*m1*m2*(x2Pred-x1Pred)/(r12**3 + epsilon)
    f12y=G*m1*m2*(y2Pred-y1Pred)/(r12**3 + epsilon)
    f13x=G*m1*m3*(x3Pred-x1Pred)/(r13**3 + epsilon)
    f13y=G*m1*m3*(y3Pred-y1Pred)/(r13**3 + epsilon)

    #corpo 2 
    f21x=-f12x
    f21y=-f12y
    f23x=G*m2*m3*(x3Pred-x2Pred)/(r23**3 + epsilon)
    f23y=G*m2*m3*(y3Pred-y2Pred)/(r23**3 + epsilon)

    #corpo 3
    f31x=-f13x
    f31y=-f13y
    f32x=-f23x
    f32y=-f23y

    #equações diferenciais do movimento para cada corpo (F=m*a) - se a rede funcionar corretamente, eq1x=eq1y=0
    eq1x=m1*ax1-(f12x+f13x)
    eq1y=m1*ay1-(f12y+f13y)
    eq2x=m2*ax2-(f21x+f23x)
    eq2y=m2*ay2-(f21y+f23y)
    eq3x=m3*ax3-(f31x+f32x)
    eq3y=m3*ay3-(f31y+f32y)

    #LOSS FÍSICA - elevo ao quadrado para evitar sinais negativos e para penalizar ainda mais os erros grandes - o .mean() retorna a média dos erros quadrados no final
    lossFisica=(eq1x**2 + eq1y**2 + eq2x**2 + eq2y**2 + eq3x**2 + eq3y**2).mean()

    #LOSS DOS DADOS - lembrando que nn.MSELoss()(previsão, dado_real) = 1/n ∑(previsão_i - dado_real_i)**2, então a MSELoss retorna o quanto os dados previstos estão distantes dos dados reais passados à função como parâmetro
    lossDados=(nn.MSELoss()(x1Pred,x1)+nn.MSELoss()(y1Pred,y1)+
               nn.MSELoss()(x2Pred,x2)+nn.MSELoss()(y2Pred,y2)+
               nn.MSELoss()(x3Pred,x3)+nn.MSELoss()(y3Pred,y3))/6
    
    #LOSS TOTAL SOMADA - 0.01 é usado quando os dados são confiáveis em precisão, 1.0 é usado quando a cinfiança dos dados é moderada (há ruído), 10.0 é usado quando os dados originais são de baiax qualidade/precisão. Quanto maior, mais a rede prioriza as leis físicas sobre os dados.
    #usarei 10.0 para peso da loss física pois os dados no arquivo NPZ têm baixíssimo ruído (ordem de grandeza entre -12 e -14)
    lossTotal=lossDados+1.0*lossFisica

    return lossTotal,lossDados,lossFisica


#instância do modelo
modelo=TresCosposPINN()
#print(modelo)


#parâmetros físicos
G=1.0
#as MASSAS já foram definidas no início do programa
epsilon=1e-8

#configuração do treinamento
#taxa de aprendizado (o quanto os dados são alterados a cada iteração para que a rede possa ter suas mutações e encontrar a versão mais adequada para descrever o fenômeno sendo estudado)
learningRate=.0005           
optimizer=t.optim.Adam(modelo.parameters(),lr=learningRate) #uso ADAM ao invés de SGD pois o erro físico pode explodir se a simulaçã for muito longa, com ordens de magnitude variadas podendo surgir, algo que o SGD não lida bem com, pois ele usa o mesmo LEARNING RATE para todos os gradientes (gradientes das losses), mas o ADAM usa learning rate adaptativo para cada parâmetro, sendo mais estável


optimizerLBFGS = t.optim.LBFGS(
    modelo.parameters(), 
    lr=1, 
    max_iter=20, 
    history_size=10, 
    line_search_fn="strong_wolfe"
)


#definindo o SCHEDULER, que vai alterar a learning rate para metade caso a loss não caia por 100 épocas. O VERBOSE vai printar uma mensagem dizendo que está diminuindo o LR para tal número
scheduler=t.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.2,patience=30,threshold=1e-5,threshold_mode='abs')


# Preparando os dados
tMin=tempoTensor.min().item()
tMax=tempoTensor.max().item()

# Treinamento 
epocas=3000
lossTotal=[]
lossDados=[]
lossFisica=[]

print("\nIniciando treinamento PINN")
print(f"Número de pontos: {len(tempoTensor)}\n")
'''print(f"Range de tempo: [{tMin:.4f}, {tMax:.4f}]")'''

#RECUPERANDO OS DADOS DA REDE NEURAL QUE EU TREINEI ATÉ AGORA

if os.path.exists("redesTreinadas/pinn3CorposCheckpointTESTE.pth"):
    checkpoint = t.load("redesTreinadas/pinn3CorposCheckpoint.pth")
    modelo.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Retomando treinamento a partir da loss: {checkpoint['loss']:.6f}")
else:
    print("Nenhum checkpoint encontrado. Começando do zero.")


for epoca in range(epocas):
    lossTotalAux, lossDadosAux, lossFisicaAux = lossPINN(
        modelo, tempoTensor, 
        x1, y1, 
        x2, y2, 
        x3, y3,
        G, m1, m2, m3, tMin, tMax
    )
    
    lossTotal.append(lossTotalAux.item())
    lossDados.append(lossDadosAux.item())
    lossFisica.append(lossFisicaAux.item())
    
    #backpropagation
    optimizer.zero_grad()
    lossTotalAux.backward()
    optimizer.step()

    scheduler.step(lossTotalAux.detach())

    if (epoca + 1) % 100 == 0:
        print(f'Época {epoca + 1:4d}, Loss total: {lossTotal[-1]:.6f}, '
              f'Loss dados: {lossDados[-1]:.6f}, '
              f'Loss física: {lossFisica[-1]:.6f}')


print("Treinamento concluído com otimizador ADAM!!!")
#refinamento com otimizador mais robusto


'''print("\nIniciando refinamento final com L-BFGS...")
tMin=tempoTensor.min().item()
tMax=tempoTensor.max().item()

modelo.train()

def closure():
    optimizerLBFGS.zero_grad()
    lTotal,_,_= lossPINN(modelo, tempoTensor, x1, y1, x2, y2, x3, y3, G, m1, m2, m3, tMin, tMax)
    lTotal.backward()
    return lTotal

# Este é um NOVO loop, independente do anterior
for i in range(50):
    loss_atual = optimizerLBFGS.step(closure)
    if (i + 1) % 10 == 0:
        print(f"L-BFGS Step {i+1}, Loss: {loss_atual.item():.6f}")

print("Refinamento concluído com otimizador L-BFGS!!!")
'''

#SALVANDO OS DADOS DA REDE TREINADA 
t.save({
    'epoca': epoca,
    #'epoca': 700,
    'model_state_dict': modelo.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    #'loss': loss_atual.item(),
    'loss': lossTotal[-1],
}, "redesTreinadas/pinn3CorposCheckpointTESTE.pth")

print("Checkpoint salvo")




# Avaliação final
modelo.eval()
with t.no_grad():
    output = modelo(tempoTensor)
    x1_pred = output[:, 0:1].numpy()
    y1_pred = output[:, 1:2].numpy()
    x2_pred = output[:, 2:3].numpy()
    y2_pred = output[:, 3:4].numpy()
    x3_pred = output[:, 4:5].numpy()
    y3_pred = output[:, 5:6].numpy()

# Convertendo para numpy para plotagem
tempo_np = tempoTensor.detach().numpy()
x1_np = x1.numpy()
y1_np = y1.numpy()
x2_np = x2.numpy()
y2_np = y2.numpy()
x3_np = x3.numpy()
y3_np = y3.numpy()

# Plotando os resultados
plt.figure(figsize=(18, 12))

# Losses
plt.subplot(2, 3, 1)
plt.semilogy(lossTotal, label='Total', linewidth=2)
plt.semilogy(lossDados, label='Dados', linewidth=2)
plt.semilogy(lossFisica, label='Física', linewidth=2)
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.title('Evolução da Loss durante treinamento')
plt.legend()
plt.grid(True)

# Corpo 1
plt.subplot(2, 3, 2)
plt.plot(x1_np, y1_np, 'b-', label='Dados reais', alpha=0.7, linewidth=1)
plt.plot(x1_pred, y1_pred, 'r--', label='PINN', alpha=0.7, linewidth=2)
plt.scatter(x1_np[0], y1_np[0], color='green', s=100, label='Início', zorder=5)
plt.scatter(x1_np[-1], y1_np[-1], color='red', s=100, label='Fim', zorder=5)
plt.xlabel('x1')
plt.ylabel('y1')
plt.title('Trajetória do Corpo 1')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Corpo 2
plt.subplot(2, 3, 3)
plt.plot(x2_np, y2_np, 'b-', label='Dados reais', alpha=0.7, linewidth=1)
plt.plot(x2_pred, y2_pred, 'r--', label='PINN', alpha=0.7, linewidth=2)
plt.scatter(x2_np[0], y2_np[0], color='green', s=100, label='Início', zorder=5)
plt.scatter(x2_np[-1], y2_np[-1], color='red', s=100, label='Fim', zorder=5)
plt.xlabel('x2')
plt.ylabel('y2')
plt.title('Trajetória do Corpo 2')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Corpo 3
plt.subplot(2, 3, 4)
plt.plot(x3_np, y3_np, 'b-', label='Dados reais', alpha=0.7, linewidth=1)
plt.plot(x3_pred, y3_pred, 'r--', label='PINN', alpha=0.7, linewidth=2)
plt.scatter(x3_np[0], y3_np[0], color='green', s=100, label='Início', zorder=5)
plt.scatter(x3_np[-1], y3_np[-1], color='red', s=100, label='Fim', zorder=5)
plt.xlabel('x3')
plt.ylabel('y3')
plt.title('Trajetória do Corpo 3')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Erro temporal
plt.subplot(2, 3, 5)
erro1 = np.sqrt((x1_np - x1_pred)**2 + (y1_np - y1_pred)**2)
erro2 = np.sqrt((x2_np - x2_pred)**2 + (y2_np - y2_pred)**2)
erro3 = np.sqrt((x3_np - x3_pred)**2 + (y3_np - y3_pred)**2)
plt.plot(tempo_np, erro1, label='Corpo 1', alpha=0.7)
plt.plot(tempo_np, erro2, label='Corpo 2', alpha=0.7)
plt.plot(tempo_np, erro3, label='Corpo 3', alpha=0.7)
plt.xlabel('Tempo normalizado')
plt.ylabel('Erro de posição')
plt.title('Erro ao longo do tempo')
plt.legend()
plt.grid(True)
plt.yscale('log')

# Comparação de trajetórias completas
plt.subplot(2, 3, 6)
plt.plot(x1_np, y1_np, 'b-', label='Real 1', alpha=0.5, linewidth=1)
plt.plot(x2_np, y2_np, 'g-', label='Real 2', alpha=0.5, linewidth=1)
plt.plot(x3_np, y3_np, 'c-', label='Real 3', alpha=0.5, linewidth=1)
plt.plot(x1_pred, y1_pred, 'r--', label='PINN 1', alpha=0.7, linewidth=2)
plt.plot(x2_pred, y2_pred, 'orange', linestyle='--', label='PINN 2', alpha=0.7, linewidth=2)
plt.plot(x3_pred, y3_pred, 'purple', linestyle='--', label='PINN 3', alpha=0.7, linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Todas as trajetórias')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()

print("\n=== Métricas Finais ===")
print(f"Loss total final: {lossTotal[-1]:.6f}")
print(f"Loss dados final: {lossDados[-1]:.6f}")
print(f"Loss física final: {lossFisica[-1]:.6f}")
print(f"Erro médio Corpo 1: {np.mean(erro1):.6f}")
print(f"Erro médio Corpo 2: {np.mean(erro2):.6f}")
print(f"Erro médio Corpo 3: {np.mean(erro3):.6f}")

#fazer uma rede classica com ann com treinamento desktop e fazer comparação entre ReLU e Sigmoide