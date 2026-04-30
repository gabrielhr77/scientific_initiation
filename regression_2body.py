#nesse código estou prevendo como a dinamica gravitacional ocorre em cooordenadas absolutas, o que é errado, pois a rede "pensa" que se eu deslocar a órbita 10 unidades à direita a situação é completamente diferente, sem levar em conta que o que importa  é a distância relativa
import numpy as np, matplotlib.pyplot as plt
import torch as tc
from torch import nn
from torch.utils.data import TensorDataset,DataLoader


#============================== SIMULAÇÃO =================================
def estadoAtual(estado,m1,m2):
    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2,t=estado

    r1=np.array([x1,y1])
    r2=np.array([x2,y2])

    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)

    a1=G*m2*(r2-r1)/r12**3
    a2=G*m1*(r1-r2)/r12**3
    
    return np.array([vx1,vy1,a1[0],a1[1],
                     vx2,vy2,a2[0],a2[1],
                     1])

def yoshida4ordem(estado,dt,m1,m2):
    w1=1/(2-2**(1/3))
    w0=-2**(1/3)/(2-2**(1/3))
    c1=c4=w1/2
    c2=c3=(w0+w1)/2
    d1=d3=w1
    d2=w0
    estado=estado.copy()

    r1=estado[0:2]
    v1=estado[2:4]
    r2=estado[4:6]
    v2=estado[6:8]

    # PRIMEIRA PARTE
    #DRIFT
    r1+=c1*dt*v1
    r2+=c1*dt*v2

    a1,a2=atualizaAceleracoes_posicoes(r1,r2,m1,m2)
    #KICK
    v1+=d1*dt*a1
    v2+=d1*dt*a2

    # SEGUNDA PARTE
    #DRIFT
    r1+=c2*dt*v1
    r2+=c2*dt*v2

    a1,a2=atualizaAceleracoes_posicoes(r1,r2,m1,m2)
    #KICK
    v1+=d2*dt*a1
    v2+=d2*dt*a2

    # TERCEIRA PARTE
    #DRIFT
    r1+=c3*dt*v1
    r2+=c3*dt*v2

    a1,a2=atualizaAceleracoes_posicoes(r1,r2,m1,m2)
    #KICK
    v1+=d3*dt*a1
    v2+=d3*dt*a2
    
    # QUARTA PARTE
    #DRIFT
    r1+=c4*dt*v1
    r2+=c4*dt*v2
    
    return np.concatenate([r1,v1,r2,v2])

def atualizaAceleracoes_posicoes(r1,r2,m1,m2):
    pr12=r2-r1
    
    rr12=pr12[0]**2 + pr12[1]**2 + epsilon**2

    inv_r12=1/(rr12*np.sqrt(rr12))

    a1=G*(m2*pr12*inv_r12)
    a2=G*(-m1*pr12*inv_r12)

    return a1,a2


def atualizaAceleracoes_estado(estado,m1,m2):
    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2,=estado

    r1=np.array([x1,y1])
    r2=np.array([x2,y2])
    
    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)
    
    a1=G*m2*(r2-r1)/r12**3
    a2=-G*m1*(r1-r2)/r12**3

    return a1,a2


def calculaEnergiaDoSistema(estado,m1,m2):
    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2=estado[:8]

    r1=np.array([x1,y1])
    r2=np.array([x2,y2])

    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)

    v1=np.array([vx1,vy1])
    v2=np.array([vx2,vy2])

    momLin.append(np.linalg.norm(m1*v1 + m2*v2))

    momAng.append(m1*(r1[0]*v1[1] - r1[1]*v1[0]) + m2*(r2[0]*v2[1] - r2[1]*v2[0]))

    cinetica=((vx1**2+vy1**2)*m1/2+(vx2**2+vy2**2)*m2/2)
    
    potencial=-G*(m1*m2/r12)
    
    r_min.append(r12)

    return cinetica+potencial

def rd(a):
    return float(a)*(2*np.random.rand()-1)


estado0=np.array([
    -10, -5, .11, .5,
     10, -10, -.2, -.6
],dtype=float)

num_simul=100

epsilon=1e-12 
dt=0.00025
steps=20000

G=1

j=0

todasTrajetorias=[]
while j<num_simul:
    print(j)
    #E_TOTAL=calculaEnergiaDoSistema
    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2=estado0

    
    m1simulacao=np.random.rand()
    m2simulacao=1.0-m1simulacao
    
    estado=np.array([rd(a) for a in[ 
        x1,y1,vx1,vy1,
        x2,y2,vx2,vy2
    ]])
    print("ESTADO0: ",estado0)
    print("MASSA ORIGINAL: ",m1simulacao,m2simulacao)
    print("\nESTADO GERADO: ",estado)
    
    

    print("\nMASSA ALTERADA: ",m1simulacao,m2simulacao,"\n")
    print("SOMA MASSAS:",m1simulacao+m2simulacao)

    r1=estado[0:2]
    v1=estado[2:4]

    r2=estado[4:6]
    v2=estado[6:8]

    P_total=m1simulacao*v1+m2simulacao*v2
    V_cm=P_total/(m1simulacao+m2simulacao)

    v1-=V_cm
    v2-=V_cm

    R_cm=(m1simulacao*r1 + m2simulacao*r2)/(m1simulacao+m2simulacao)

    r1-=R_cm
    r2-=R_cm

    simulacaoAtual=[]#trajetoria atual da simulação
    tempoSimulacao=[]
    energiaDoSistema=[]
    r_min=[]
    momLin=[]
    momAng=[]

    passoAtual=0
    flag_colisao=0
    tAtual=0.0

    for i in range(steps):
        passoAtual+=1
        estadoComTempo=np.append(estado.copy(),tAtual)
        simulacaoAtual.append(estadoComTempo)#aqui o trajetoria é uma lista de arrays
        tempoSimulacao.append(tAtual)

        energiaDoSistema.append(calculaEnergiaDoSistema(estado,m1simulacao,m2simulacao))
        
        estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao)

        r12=np.sqrt((np.linalg.norm(np.array([estado[4],estado[5]])-np.array([estado[0],estado[1]])))**2 + epsilon**2)
        
        if passoAtual % 10000 == 0 and passoAtual > 0:
            print(f"Passo {passoAtual}/{steps} ({100*passoAtual/steps:.1f}%)")
        if r12<0.2:
            print("COLISAO DETECTADA, CANCELANDO SIMULAÇÃO")
            print("\nPASSO \n",passoAtual)
            flag_colisao=1
            break
        
    if(flag_colisao==0): 
        trajetoria=np.array(simulacaoAtual)
        todasTrajetorias.append({
            'dados': np.array(simulacaoAtual),
            'm1': m1simulacao,
            'm2': m2simulacao
        })
        j+=1

        

        x1=trajetoria[:,0]
        y1=trajetoria[:,1]
        vx1=trajetoria[:,2]
        vy1=trajetoria[:,3]
        x2=trajetoria[:,4]
        y2=trajetoria[:,5]
        vx2=trajetoria[:,6]
        vy2=trajetoria[:,7]

        '''#==============================GERAÇÃO DE PLOTS EM UMA MESMA IMAGEM==============================
        fig, axs = plt.subplots(3, 2, figsize=(12, 12))
        # ---------------- TRAJETÓRIA ----------------
        axs[0,0].plot(x1, y1, label='corpo 1')
        axs[0,0].plot(x2, y2, label='corpo 2')
        axs[0,0].scatter(x1[0], y1[0])
        axs[0,0].scatter(x2[0], y2[0])
        axs[0,0].set_title("Trajetória")
        axs[0,0].axis("equal")
        axs[0,0].grid()
        # ---------------- r_min ----------------
        axs[0,1].plot(r_min)
        axs[0,1].set_title("Distância mínima")
        axs[0,1].grid()
        # ---------------- Momento Linear ----------------
        axs[1,0].plot(momLin)
        axs[1,0].set_title("Momento Linear")
        axs[1,0].grid()
        # ---------------- Momento Angular ----------------
        axs[1,1].plot(momAng)
        axs[1,1].set_title("Momento Angular")
        axs[1,1].grid()
        # ---------------- Energia ----------------
        axs[2,0].plot(energiaDoSistema)
        axs[2,0].set_title("Energia Total")
        axs[2,0].grid()
        # ---------------- Erro relativo ----------------
        E0 = energiaDoSistema[0]
        erro_relativo = (energiaDoSistema - E0)/abs(E0)
        axs[2,1].plot(erro_relativo)
        axs[2,1].set_title("Erro Relativo da Energia")
        axs[2,1].grid()

        plt.tight_layout()
        plt.show()
        print(f"\nSIMULACAO NUMERO: {j+1}\n")

        print("\nSE DESEJAR OUTRA SIMULACAO DIGITE 0, SE ESTA JÁ ESTÁ DE BOM TAMANHO, DIGITE 1")
        validacao=int(input())
        if validacao==0: j-=1'''
    




#============================== REDE =================================
'''coordsMax=max(x1.max(),y1.max(),x2.max(),y2.max())
x1=x1/coordsMax
y1=y1/coordsMax
x2=x2/coordsMax
y2=y2/coordsMax'''#aqui apenas dividir pelo maximo não é boa normalização, vou ter que fazer a Z-SCORE

M=10
alpha=1e-8



X=[]#valor inicial
Y=[]#vai ser o resultado final da rede almejado

k=10

#gerando o vetor de inputs e de alvos para cada input
for traj in todasTrajetorias:
    trajetoriaGerada=traj['dados']
    m1=traj['m1']
    m2=traj['m2']
    for i in range(len(traj['dados'])-200):
        #k=np.random.randint(1,20)
        
        estadoT=np.concatenate([trajetoriaGerada[i,:8],[m1,m2,k*dt]])
        #estadoTmaist=trajetoria[i+k,:8] #prevendo o proximo estado
        delta=trajetoriaGerada[i+k,:8]-trajetoriaGerada[i,:8] #prevendo o incremento

        X.append(estadoT)
        Y.append(delta)

Y=np.array(Y)
X=np.array(X)

#entrada e saída da rede
X_media=X.mean(axis=0)
X_std=X.std(axis=0)+1e-8
Y_media=Y.mean(axis=0)
Y_std=Y.std(axis=0)+1e-8

#normalizando com Z-SCORE
X_normal=(X-X_media)/X_std
Y_normal=(Y-Y_media)/Y_std

#convertendo para tensores - preparando o dataset
X_tensor=tc.tensor(X_normal,dtype=tc.float32)
Y_tensor=tc.tensor(Y_normal,dtype=tc.float32)

dataSet=TensorDataset(X_tensor,Y_tensor)

print('.')


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
            nn.Linear(11,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,8)
        )
        '''self.net=nn.Sequential(
            nn.Linear(11,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,8)
        )'''
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


#trainLoader=DataLoader(dataSet,batch_size=128,shuffle=True)
trainLoader=DataLoader(dataSet,batch_size=64,shuffle=True)
print('..') 

historico_loss=[]
#rodando o loop de treino
#for i in range(num_simul):
for i in range(3):
    for epoca in range(5):
        print('\n-----------------------------------------------')
        print(f'Comecando epoca {epoca+1}')
        loss_epoca=0.0
        for inputs,alvos in trainLoader:
            predicao=modelo(inputs)
            loss=loss_padrao(predicao,alvos)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoca+=loss.item()
            
        loss_medio=loss_epoca/len(trainLoader)
        historico_loss.append(loss_medio)
        print(f'Epoca {epoca+1}: {loss_medio}')

m1_validacao=0.25 
m2_validacao=0.75
estado_validacao=np.array([5.0,0.0,0.0,0.5,-5.0,0.0,0.0,-0.2])

trajetoria_validacao = []
temp_estado = estado_validacao.copy()
for _ in range(steps):
    trajetoria_validacao.append(np.append(temp_estado.copy(), 0)) # t=0 simplificado
    temp_estado = yoshida4ordem(temp_estado, dt, m1_validacao, m2_validacao)

trajetoria_validacao = np.array(trajetoria_validacao)
"""teste=tc.tensor([[
-1.0,0.0,0.1,0.3,
1.0,0.0,-0.1,-0.3,
0.4,0.6,
50*dt
]],dtype=tc.float32)
print(f'TESTE COM VALOR DESCONHECIDO: {modelo(teste)}') #predição normalizada da rede [ax_pred,ay_pred]

#GERANDO UMA SIMULAÇÃO NOVA PARA TESTAR SE A REDE APRENDEU MESMO"""


passo_inicial = 0
k_teste = k
#horizonte = 5000
#trajetoriaTeste=yoshida4ordem(teste,dt,m1,m2)
horizonte = (len(trajetoria_validacao)-100)//k_teste
delta_t = k_teste*dt

# estado inicial real
estado_real = trajetoria_validacao[passo_inicial,:8].copy()

# mesma condição inicial para rede
estado_pred = estado_real.copy()


traj_real_1=[]
traj_real_2=[]

traj_pred_1=[]
traj_pred_2=[]

#loop de teste da rede
for n in range(horizonte):
    entradaBruta=np.concatenate([estado_pred,[m1_validacao,m2_validacao,delta_t]])
    entradaNormal=(entradaBruta-X_media)/X_std
    entradaTensor=tc.tensor([entradaNormal],dtype=tc.float32)

    estado_real = trajetoria_validacao[passo_inicial+n*k_teste,:8]

    traj_real_1.append(
        [estado_real[0],estado_real[1]]
    )

    traj_real_2.append(
        [estado_real[4],estado_real[5]]
    )


    # -----trajetória gerada pela rede-----
    entrada=np.concatenate([
        estado_pred,
        [m1_validacao,m2_validacao,delta_t]
    ])

    entrada=tc.tensor(
        [entrada],
        dtype=tc.float32
    )

    with tc.no_grad():
        delta_normal = modelo(entradaTensor).numpy()[0]

    deltaNaoNormal=(delta_normal*Y_std)+Y_media
    estado_pred = estado_pred + deltaNaoNormal

    traj_pred_1.append(
        [estado_pred[0],estado_pred[1]]
    )

    traj_pred_2.append(
        [estado_pred[4],estado_pred[5]]
    )


traj_real_1=np.array(traj_real_1)
traj_real_2=np.array(traj_real_2)

traj_pred_1=np.array(traj_pred_1)
traj_pred_2=np.array(traj_pred_2)

#============================plot final com resultados=============================
plt.figure(figsize=(10,10))

# Corpo 1 real
plt.plot(
traj_real_1[:,0],
traj_real_1[:,1],
label="Corpo1 Simulação"
)

# Corpo1 rede
plt.plot(
traj_pred_1[:,0],
traj_pred_1[:,1],
'--',
label="Corpo1 Rede"
)


# Corpo2 real
plt.plot(
traj_real_2[:,0],
traj_real_2[:,1],
label="Corpo2 Simulação"
)

# Corpo2 rede
plt.plot(
traj_pred_2[:,0],
traj_pred_2[:,1],
'--',
label="Corpo2 Rede"
)

plt.scatter(
traj_real_1[0,0],
traj_real_1[0,1],
s=80
)

plt.scatter(
traj_real_2[0,0],
traj_real_2[0,1],
s=80
)

plt.legend()
plt.grid()
plt.axis("equal")
plt.title(
"Simulação Yoshida vs Rede Neural"
)
plt.show()

plt.plot(historico_loss)
'''plt.set_yscale('log')
plt.set_title("Curva de aprendizado")
plt.set_xlabel("Épocas")
plt.set_ylabel("Loss (MSE)")'''

plt.tight_layout() 
#plt.show()