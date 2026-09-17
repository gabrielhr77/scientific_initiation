import numpy as np
import torch as t
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
import re #biblioteca que vai me permitir retomar o treinamento de uma certa época, se ter que treinar tudo do zero sempre
from tqdm import tqdm #biblioteca para visualizar avanço do treinamento

G=1.0
epsilon=1e-6

# ================================================================ DADOS ================================================================
def carregarTrajetorias(pastaNPZ):
    trajetorias,massasLista=[],[]
    for arquivo in sorted(Path(pastaNPZ).glob("*.npz")):    #pega um arquivo .npz aleatório para termos mais uma camada de aleatoriedade
        dados = np.load(arquivo)
        estado = dados["estado"]
        if estado.ndim < 2 or estado.shape[0] == 0:         #verificação de tamanho inválido --> caso o arquivo tenha a primeira coluna sendo igual a zero isso indica que a simulação foi falha ou já começou em uma colisão ou que ela ocorreu logo nos primeiros passos
            print(f"[AVISO] pulando {arquivo.name}: trajetória vazia/colisão premtura demais para Yoshida gerar dados válidos")
            continue
        estado=estado[:,:18]                             #pega apenas os dados que importam --> posições e velocidades
        trajetorias.append(estado)
        massasLista.append(dados["massas"])
    return trajetorias,massasLista

def acharUltimoCheckpoint(pasta="."):
    checkpoints=sorted(Path(pasta).glob("checkpointEpoca*.pt"))
    if not checkpoints:
        return None,0
    ultimo=checkpoints[-1]
    epoca=int(re.search(r"checkpointEpoca(\d+)\.pt",ultimo.name).group(1))
    return ultimo, epoca+1

def calcularNormalizacao(trajetorias):
    todos=np.concatenate(trajetorias)
    media=todos.mean(axis=0)
    desvio=todos.std(axis=0)
    desvio[desvio<1e-8]=1.0                             #verifica dentro do vetor desvio e substitui os valores que são menores que 1e-8 por 1 (evita que haja divisão por zero em calculos posteriores)
    return media, desvio

def normalizar(x, media, desvio):                       #famoso Z-SCORE
    #forço tudo que chega para TENSOR para que o numpy não tente transformar o tensor em array numpy, o que quebraria o grafo e tornaria o cálculo do gradiente inconsistente
    x=t.as_tensor(x,dtype=t.float32)
    media=t.as_tensor(media,dtype=t.float32,device=x.device)
    desvio=t.as_tensor(desvio,dtype=t.float32,device=x.device)
    return (x - media) / desvio

def desnormalizar(x, media, desvio):                    #retorna para os dados originais
    x=t.as_tensor(x,dtype=t.float32)
    media=t.as_tensor(media,dtype=t.float32,device=x.device)
    desvio=t.as_tensor(desvio,dtype=t.float32,device=x.device)
    return x * desvio + media

class JanelaComHorizonte(Dataset):
    #aqui tenho o HORIZONTE, que é o tamanho máximo de passos futuros que vou querer que a rede aprenda a prever dado o treinamento com a janela passada --> exemplo abaixo
    # seja o vetor original dos estados [t0, t1, t2, t3, t4, t5, t6, t7, t8] e tamanhoJanela=6 e horizonteMax=3
    # entrego [t0, t1, t2, t3, t4, t5] e quero que a rede me retorne [t6, t7, t8]
    def __init__(self,trajetorias,massas,tamanhoJanela,horizonteMax,media,desvio,mediaMassa,desvioMassa):
        self.tamanhoJanela=tamanhoJanela
        self.horizonteMax=horizonteMax
        self.trajetorias=[normalizar(traj,media,desvio) for traj in trajetorias]
        self.massas=[normalizar(m,mediaMassa,desvioMassa) for m in massas]

        comprimentoTotal=tamanhoJanela+horizonteMax
        self.indices=[]
        for idTraj,traj in enumerate(self.trajetorias):        #enumerate --> mantém uma tupla (id, valor) ao iterar sobre um iterável como um vetor ou list
            nAmostras=len(traj)-comprimentoTotal+1
            pulo=10                                             #tenho que usar isso pois estava gerando muitas janelas com valores identicos nelas (se a primeira tinha do 1 ao 15, a segunda tinha do 2 ao 16, sendo que cada passo é um incremento pequeno para chegar ao outro, não mudando quase nada entre dois passos), por isso adicionei um pulo para ter menos janelas
            for i in range(0, max(nAmostras,0), pulo):
                self.indices.append((idTraj,i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,idx):
        idTraj,i=self.indices[idx]     #idTraj é qual trajetória será usada, i é qual o índice dentro daquela trajetória será usado para definir a janela em questão
        traj=self.trajetorias[idTraj]
        janela=traj[i:i+self.tamanhoJanela]
        futuros=traj[i+self.tamanhoJanela : i+self.tamanhoJanela+self.horizonteMax]     #pega a partir do indice i+tamanhoJanela até isso mais o horizonte determinado
        massa=self.massas[idTraj]
        return (janela.clone(),massa.clone(),futuros.clone())

class JanelaProximoEstado(Dataset):
#usado para pegar a janela de validação --> testar a predição da rede e verificar o loss de validação 
    def __init__(self,trajetorias,massas,tamanhoJanela,media,desvio,mediaMassa,desvioMassa):
        self.tamanhoJanela=tamanhoJanela
        self.trajetorias=[normalizar(traj,media,desvio) for traj in trajetorias]
        self.massas=[normalizar(m,mediaMassa,desvioMassa) for m in massas]

        self.indices=[]
        for idTraj,traj in enumerate(self.trajetorias):
            nAmostras=len(traj)-tamanhoJanela
            pulo=10                                             #tenho que usar isso pois estava gerando muitas janelas com valores identicos nelas (se a primeira tinha do 1 ao 15, a segunda tinha do 2 ao 16, sendo que cada passo é um incremento pequeno para chegar ao outro, não mudando quase nada entre dois passos), por isso adicionei um pulo para ter menos janelas
            for i in range(0, max(nAmostras,0), pulo):
                self.indices.append((idTraj,i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idTraj, i = self.indices[idx]
        traj = self.trajetorias[idTraj]
        janela = traj[i:i + self.tamanhoJanela]
        alvo = traj[i + self.tamanhoJanela]
        massa = self.massas[idTraj]
        return (janela.clone(),massa.clone(),alvo.clone())


# ================================================================ FÍSICA ================================================================
def calcularHamiltoniano(estado, massas):    
    r1=estado[...,0:3]
    v1=estado[...,3:6]
    r2=estado[...,6:9]
    v2=estado[...,9:12]
    r3=estado[...,12:15]
    v3=estado[...,15:18]
    #a maneira de baixo quebra pois estou usando batches, e usar simplesmente os 3 primeiros elementos do vetor estado não é correto visto qe não é mais (18,), mas sim (B,18)
    '''r1=estado[0:3]
    v1=estado[3:6]
    r2=estado[6:9]
    v2=estado[9:12]
    r3=estado[12:15]
    v3=estado[15:18]'''
    m1,m2,m3=massas[...,0],massas[...,1],massas[...,2]
    r12=t.sqrt(t.linalg.norm(r2-r1, dim=-1)**2+epsilon**2)
    r13=t.sqrt(t.linalg.norm(r3-r1, dim=-1)**2+epsilon**2)
    r23=t.sqrt(t.linalg.norm(r3-r2, dim=-1)**2+epsilon**2)

    energiaCineticaSistema=(0.5*m1*t.sum(v1**2, dim=-1)+0.5*m2*t.sum(v2**2, dim=-1)+0.5*m3*t.sum(v3**2, dim=-1))                 
    energiaPotencialSistema=-G*(m1*m2/r12 + m1*m3/r13 + m2*m3/r23)

    return energiaCineticaSistema+energiaPotencialSistema
    
def calcularMomentoLinear(estado, massas):      #IMPORTADO DE fisica3corpos.py
    v1=estado[...,3:6]
    v2=estado[...,9:12]
    v3=estado[...,15:18]
    m1,m2,m3=massas[...,0],massas[...,1],massas[...,2]
    #aqui estava dando erro de BROADCASTING pois o massas está com (B,) e velocidades estão com (B,3)
    #return t.linalg.norm(m1*v1+m2*v2+m3*v3)
    #aqui preciso fazer um insqueeze(-1)
    #já o dim=-1 faz com que o os cálculos de NORMA e PRODUTO VETORIAL do torch não façam cálculos com todos os tensores (dos vários batches), mas sim apenas com o tensor do batch em questão (tal tensor é o último componente do tensor total (B,K,3), logo quando coloco dim=-1 eu pego apenas a última coluna deste tensor)
    return t.linalg.norm((m1[...,None]*v1 + m2[...,None]*v2 + m3[...,None]*v3),dim=-1)

def calcularMomentoAngular(estado, massas):      #IMPORTADO DE fisica3corpos.py
    r1=estado[...,0:3]
    v1=estado[...,3:6]
    r2=estado[...,6:9]
    v2=estado[...,9:12]
    r3=estado[...,12:15]
    v3=estado[...,15:18]
    m1,m2,m3=massas[...,0],massas[...,1],massas[...,2]
    return m1[...,None]*t.cross(r1,v1,dim=-1)+m2[...,None]*t.cross(r2,v2,dim=-1)+m3[...,None]*t.cross(r3,v3,dim=-1)

def atualizaAceleracoes_posicoes3Corpos(r1,r2,r3,m1,m2,m3):
    #vetor posição
    pr12=r2-r1
    pr13=r3-r1
    pr23=r3-r2
    #radicando
    '''rr12=pr12[0]**2 + pr12[1]**2 + pr12[2]**2 + epsilon**2
    rr13=pr13[0]**2 + pr13[1]**2 + pr13[2]**2 + epsilon**2
    rr23=pr23[0]**2 + pr23[1]**2 + pr23[2]**2 + epsilon**2'''
    rr12=t.sum(pr12**2,dim=-1)+epsilon**2
    rr13=t.sum(pr13**2,dim=-1)+epsilon**2
    rr23=t.sum(pr23**2,dim=-1)+epsilon**2

    #fazendo o inverso para poupar algumas divisões
    inv_r12=1/(rr12*t.sqrt(rr12))
    inv_r13=1/(rr13*t.sqrt(rr13))
    inv_r23=1/(rr23*t.sqrt(rr23))

    a1=G*(m2[...,None]*pr12*inv_r12[...,None] + m3[...,None]*pr13*inv_r13[...,None])
    a2=G*(-m1[...,None]*pr12*inv_r12[...,None] + m3[...,None]*pr23*inv_r23[...,None])
    a3=G*(-m1[...,None]*pr13*inv_r13[...,None] - m2[...,None]*pr23*inv_r23[...,None])

    return a1,a2,a3

def integradorYoshida4aOrdem(estado,dt,m1,m2,m3): 
    w1=1/(2-2**(1/3))
    w0=-2**(1/3)/(2-2**(1/3))
    c1=c4=w1/2
    c2=c3=(w0+w1)/2
    d1=d3=w1
    d2=w0
    #estado=estado.clone()

    r1=estado[...,0:3]
    v1=estado[...,3:6]
    r2=estado[...,6:9]
    v2=estado[...,9:12]
    r3=estado[...,12:15]
    v3=estado[...,15:18]

    # PRIMEIRA PARTE
    #DRIFT
    r1=r1+c1*dt*v1
    r2=r2+c1*dt*v2
    r3=r3+c1*dt*v3

    a1,a2,a3=atualizaAceleracoes_posicoes3Corpos(r1,r2,r3,m1,m2,m3)
    #KICK
    v1=v1+d1*dt*a1
    v2=v2+d1*dt*a2
    v3=v3+d1*dt*a3

    # SEGUNDA PARTE
    #DRIFT
    r1=r1+c2*dt*v1
    r2=r2+c2*dt*v2
    r3=r3+c2*dt*v3

    a1,a2,a3=atualizaAceleracoes_posicoes3Corpos(r1,r2,r3,m1,m2,m3)
    #KICK
    v1=v1+d2*dt*a1
    v2=v2+d2*dt*a2
    v3=v3+d2*dt*a3

    # TERCEIRA PARTE
    #DRIFT
    r1=r1+c3*dt*v1
    r2=r2+c3*dt*v2
    r3=r3+c3*dt*v3

    a1,a2,a3=atualizaAceleracoes_posicoes3Corpos(r1,r2,r3,m1,m2,m3)
    #KICK
    v1=v1+d3*dt*a1
    v2=v2+d3*dt*a2
    v3=v3+d3*dt*a3

    # QUARTA PARTE
    #DRIFT
    r1=r1+c4*dt*v1
    r2=r2+c4*dt*v2
    r3=r3+c4*dt*v3

    return t.cat([r1,v1,r2,v2,r3,v3],dim=-1)

# ================================================================ ARQUITETURA DA REDE ================================================================
class EmbeddingFisico(t.nn.Module):
    def __init__(self, dimEstado=18, dModel=64, bias=True):
        super().__init__()
        self.proj=t.nn.Linear(dimEstado,dModel,bias=bias)

    def forward(self, janela):
        return self.proj(janela)

class CodificacaoPosicional(t.nn.Module):   
    def __init__(self, dModel, tamanhoJanelaMax=50):
        super().__init__()
        #o t.nn.Embedding cria uma tabela de vetores individuais treináveis onde cada linha dessa matriz é o vetor de posição que foi aprendido para o passo i da janela
        #vale lembrar que os valores iniciais são aleatórios, mas que o treino vai ajustando via BACKPROPAGATION
        self.embeddPosicao=t.nn.Embedding(tamanhoJanelaMax,dModel)    
        
    def forward(self, i):
        # B --> tamanho do batch (quantidade de janelas)
        # K --> quantos passos tem essa janela
        # _ --> é o "comprimento" do DMODEL, ou seja, a quantidade de elementos do vetor traduzido do real vetor de posição
        #aqui leva em consideração que, caso B=2, K=3 e dModel=4, i teria shape=(2,3,4) e seria algo como
        #janela 1:
        #  passo 0: [1.0, 1.0, 1.0, 1.0]
        #  passo 1: [2.0, 2.0, 2.0, 2.0]
        #  passo 2: [3.0, 3.0, 3.0, 3.0]

        #janela 2:
        #  passo 0: [9.0, 9.0, 9.0, 9.0]
        #  passo 1: [8.0, 8.0, 8.0, 8.0]
        #  passo 2: [7.0, 7.0, 7.0, 7.0]
        
        B,K,_=i.shape
        posicoes=t.arange(K,device=i.device)    #cria o vetor com as posições de cada pasos da janela, já o i.device joga para a GPU ou CPU, dependendo de onde está i
        #self.embeddPosicao --> matriz de numeros que o modelo vai aprender
        posEmb=self.embeddPosicao(posicoes) #pega as linhas presentes no vetor posições      
        #como i é de 3 dimensões (2,3,4) e posicoes é de 2 dimensões (3,4) o .unsqueeze(0) apenas adiciona uma dimensão de tamanho 1 na frente, tornando posEmb (1,3,4)
        #posEmb antes do .unesqueeze(0): [ [0.1,0.2,0.3,0.4], [0.5,0.1,0.9,0.2], [0.3,0.7,0.1,0.6] ] --> 3 batches de 4 passos
        #posEmb após o .unesqueeze(0): [ [ [0.1,0.2,0.3,0.4], [0.5,0.1,0.9,0.2], [0.3,0.7,0.1,0.6] ] ] --> 1 batch de 3 passos
        #aqui é onde ocorre aquela parte do Transformer onde ele entende a posição de cada passo da janela, essa soma
        #gera (2,3,4)+(1,3,4), e quando uma das dimensões é 1 o PyTorch repete aquele dado para bater com o outro tamanho (famoso BROADCASTNG) 
        #ou seja, janela 1, passo 0: [1, 1, 1, 1] + [0.1,0.2,0.3,0.4] = [1.1,1.2,1.3,1.4]
        #janela 1, passo 1: [2, 2, 2, 2] + [0.5,0.1,0.9,0.2] = [2.5,2.1,2.9,2.2]
        #assim o Transformer consegue compreender sequencia nas janelas
        return i+posEmb.unsqueeze(0)

class RedeTransformer(t.nn.Module):
    def __init__(self, dimEstado=18, dimMassa=3, dModel=64, nHeads=4,
                 nCamadas=6, dimFeedforward=256, dropout=0.1, tamanhoMaximoJanela=50):
        super().__init__()
        self.dimEstado=dimEstado
        self.dimMassa=dimMassa
        self.embedding=EmbeddingFisico(dimEstado+dimMassa,dModel)
        self.codPosicional=CodificacaoPosicional(dModel,tamanhoMaximoJanela)
        self.dropoutEntrada=t.nn.Dropout(dropout)   #técnica de regularização, zerando aleatoriamente uma porcentagem referente ao "dropout" --> isso é feito para que a rede não decore o treino, mas sim aprenda os pad~roes gerais (a rede não vai depender demais de um neurônio específico ao implementar isso)
        #aqui o "TransformerEncoderLayer" faz, mais ou menos, tensor -> multi-head self-attention -> Add e normalização -> feedforward -> Add e normalização -> saída
        # d_model é a dimensão dos vetores que entram e saem, nhead é a quantidade de cabeças de atenção paralelas que irão prestar atenção em diferentes pontos da sequencia (tem que ser um numero que possa dividir igualmente o dModel 64/4)
        #dim_feedforward é o tamanho da camda oculta onde a rede vai processar 
        #dropout vai aplicar internamente nessa camada o dropout na atenção e no feedforward
        #como falado no stackoverflow, tenho que manter batch_first=True pois senão o PyTorch vai usar o padrão antigo de (K -sequencia-, B -batch-, dModel), e não o que estou usando aqui (B, K, dModel)
        camada=t.nn.TransformerEncoderLayer(d_model=dModel,nhead=nHeads,dim_feedforward=dimFeedforward,dropout=dropout,batch_first=True)
        #agora faço nCamadas cópias independentes com seus próprios pesos dessa camada que configurei acima 
        self.encoder=t.nn.TransformerEncoder(camada,num_layers=nCamadas)
        #agora retorno de dModel para a língua original de posições e velocidades (retraduzo)
        self.cabecaSaida=t.nn.Linear(dModel,dimEstado)


    def forward(self, janela, massa):
        B,K,_=janela.shape

        #coloco a massa em cada passo da janela para que a rede aorenda que há relação entre massa e a situação atual do sistema/da janela
        #uso o unsqueeze para adicionar uma dimensão e poder concatenar com o estado
        massaExpandida=massa.unsqueeze(1).expand(B,K,self.dimMassa)

        #concateno o estado com a massa em cada passo
        entradaRede=t.cat([janela,massaExpandida],dim=-1)

        #faço o embedding físico, a condificaçãpo posicional e o dropout para evitar decoreba
        i=self.embedding(entradaRede)
        i=self.codPosicional(i)
        i=self.dropoutEntrada(i)

        #fase do self-attention, feedforward, add norm e N camadas
        saida=self.encoder(i)

        #pego a ultima posição da janela
        ultimaPosicao=saida[:, -1, :]

        #retraduzo
        proximoEstado=self.cabecaSaida(ultimaPosicao)

        return proximoEstado
    
# ================================================================ LOSS ================================================================
def lossEstado(previsto, real):
    #melhor usar nn.functional.mse_loss pois nao precisa instanciar a classe como em nn.MSELoss
    return t.nn.functional.mse_loss(previsto, real)

def lossEnergia(previsto, real, massas):
    #retorna (H(real) - H(previsto))^2
    return (calcularHamiltoniano(real,massas)-calcularHamiltoniano(previsto,massas))**2

def lossMomentoLinear(previsto, real, massas):
    return (calcularMomentoLinear(previsto,massas)-calcularMomentoLinear(real,massas))**2

def lossMomentoAngular(previsto, real, massas):
    return (t.linalg.norm(calcularMomentoAngular(previsto,massas)-calcularMomentoAngular(real,massas),dim=-1))**2

def lossConsistenciaYoshida(previsto, ultimoEstadoJanela, massas, dt):
    return (t.linalg.norm(previsto-integradorYoshida4aOrdem(ultimoEstadoJanela,dt,massas[...,0],massas[...,1],massas[...,2]),dim=-1))**2
    
def lossTotal(previsto, real, ultimoEstadoJanela, massas, dt, pesos):
    termosDaLOSS = {
        "estado": lossEstado(previsto, real),
        "energia": lossEnergia(previsto, real, massas),
        "momentoLinear": lossMomentoLinear(previsto, real, massas),
        "momentoAngular": lossMomentoAngular(previsto, real, massas),
        "consistencia": lossConsistenciaYoshida(previsto, ultimoEstadoJanela, massas, dt),
    }
    total = sum(pesos[nome] * valor for nome, valor in termosDaLOSS.items())  #pega cada par de (nome,valor) e itera, passando para TOTAL o valor da soma de todas as LOSSes
    return total, termosDaLOSS

# ================================================================ DIAGNOSTICOS ================================================================
def baselineDePersistencia(loaderValidacao):
    criterio = t.nn.MSELoss()
    lossAcumulada = 0.0
    for janela, _massa, alvo in loaderValidacao:
        predPersistencia = janela[:, -1, :]
        lossAcumulada += criterio(predPersistencia, alvo).item()
    return lossAcumulada / len(loaderValidacao)

def horizontePrevisibilidade(real, previsto):
    escalaReferencia = np.linalg.norm(real - real.mean(axis=0), axis=1).mean()
    erro = np.linalg.norm(real - previsto, axis=1)
    acimaDoLimiar = np.where(erro > escalaReferencia)[0]
    horizonte = acimaDoLimiar[0] if len(acimaDoLimiar) > 0 else len(erro)
    return horizonte, escalaReferencia

def probabilidadeTeacherForcing(epoca,k=5.0):
    # fica proximo de 1 no inicio do treino, mas decresce rápido no final, 
    # fazendo com que a rede, após já estar mais precisa, possa aprender como seus pequenos erros podem ser consertados
    # o K é a velocidade de decaimento
    # 𝜎(𝑥)=1/(1+e^(-x))
    p = k / (k + math.exp(epoca / k))
    return max(0.0, min(1.0, p))

# ================================================================ AVALIAÇÃO ================================================================
def rolloutAutoregressivo(modelo,trajetoriaNormalizada,massaNormalizada,tamanhoJanela, nPassos,device):
    modelo.eval()
    janelaAtual=trajetoriaNormalizada[:tamanhoJanela].to(device).unsqueeze(0) 
    massa=massaNormalizada.to(device).unsqueeze(0) 
    previsoes=[]
    with t.no_grad():   #sem gradiente, apenas usando os valores, que é o que realmente impota aqui
        for _ in range(nPassos):
            proximo=modelo(janelaAtual, massa)
            previsoes.append(proximo.squeeze(0).cpu().numpy())
            janelaAtual=t.cat([janelaAtual[:, 1:, :],proximo.unsqueeze(1)],dim=1)
    return np.array(previsoes)

def avaliarRollout(modelo,trajetoriasValidacao,massasValidacao,tamanhoJanela,media,desvio,mediaMassa,desvioMassa,device,nTrajetoriasMax=12,nPassosDoRollout=300):
    modelo.eval()
    nTraj = min(len(trajetoriasValidacao), nTrajetoriasMax)

    fig, eixos = plt.subplots(4, 3, figsize=(18, 20))
    eixos = eixos.flatten()

    horizontes =[]
    errosMedios=[]
    for idx in range(nTraj):
        trajReal=trajetoriasValidacao[idx]
        massaReal=massasValidacao[idx]       
        nPassos=min(nPassosDoRollout, len(trajReal) - tamanhoJanela)
        if nPassos <= 0:
            continue
        trajNorm=normalizar(trajReal,media,desvio)
        massaNorm=normalizar(massaReal,mediaMassa,desvioMassa)

        previstoNorm = rolloutAutoregressivo(modelo,trajNorm,massaNorm,tamanhoJanela,nPassos,device)
        previstoReal = desnormalizar(previstoNorm, media, desvio).numpy()   #tenho que passar pra numpy pois os plots precisam estar em numpy e não e tensor
        alvoReal = trajReal[tamanhoJanela : tamanhoJanela + nPassos]

        erro = np.linalg.norm(previstoReal - alvoReal, axis=1)
        erroMedio = erro.mean()
        errosMedios.append(erroMedio)

        horizonte, escalaRef = horizontePrevisibilidade(alvoReal, previstoReal)
        horizontes.append(horizonte)

        ax = eixos[idx]
        passos = np.arange(nPassos)
        ax.plot(passos, alvoReal[:, 0], label="Real (corpo 1, X)")
        ax.plot(passos, previstoReal[:, 0], '--', label="Predito (corpo 1, X)")
        ax.set_title(f"Traj {idx} | erro médio: {erroMedio:.3f} | horiz.: {horizonte}")
        ax.set_xlabel("Passo")
        ax.set_ylabel("Posição X")
        if idx == 0:
            ax.legend()
        ax.grid(True)

    for j in range(nTraj, len(eixos)):
        eixos[j].axis('off')

    plt.suptitle(f"Rollout autoregressivo — validação "
                f"(mostrando {nTraj} de {len(trajetoriasValidacao)} trajetórias)")
    plt.tight_layout()
    plt.savefig("rollout_validacao.png", dpi=120)
    plt.show()

    print(f"\nErro médio entre trajetórias ({nTraj} avaliadas): {np.mean(errosMedios):.4f}")
    print(f"Desvio padrão entre trajetórias: {np.std(errosMedios):.4f}")
    print(f"Horizonte de previsibilidade — média: {np.mean(horizontes):.2f} | "
        f"mín/máx: {np.min(horizontes)}/{np.max(horizontes)}")

    return horizontes,errosMedios

# ================================================================ TREINO ================================================================
if __name__ == "__main__":
    # ------------------- Hiperparâmetros -----------------------------
    TAMANHO_JANELA = 10
    D_MODEL = 64
    N_HEADS = 4
    N_CAMADAS = 6
    DIM_FEEDFORWARD = 256
    DROPOUT = 0.1
    HORIZONTE_MAX = 5
    N_EPOCAS = 10
    DT = 0.00025                     
    PESOS_LOSS = {
        "estado": 1.0,
        "energia": 0.01,
        "momentoLinear": 0.01,
        "momentoAngular": 0.01,
        "consistencia": 0.1,
    }

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"Rodando em: {device}")
    # ------------------- Dados ----------------------
    trajetorias, massas = carregarTrajetorias("simulacoesArtificiais/simulacoes3C")

    nValidacao = max(1, len(trajetorias) // 5)
    trajetoriasTreino, trajetoriasValidacao = trajetorias[nValidacao:], trajetorias[:nValidacao]
    massasTreino, massasValidacao = massas[nValidacao:], massas[:nValidacao]

    media, desvio = calcularNormalizacao(trajetoriasTreino)
    mediaMassa = np.mean(massasTreino, axis=0)
    desvioMassa = np.std(massasTreino, axis=0)
    desvioMassa = np.where(desvioMassa < 1e-8, 1.0, desvioMassa)

    datasetTreino = JanelaComHorizonte(trajetoriasTreino, massasTreino, TAMANHO_JANELA, HORIZONTE_MAX,
                                        media, desvio, mediaMassa, desvioMassa)
    datasetValidacao = JanelaProximoEstado(trajetoriasValidacao, massasValidacao, TAMANHO_JANELA,
                                            media, desvio, mediaMassa, desvioMassa)

    '''loaderTreino = DataLoader(datasetTreino, batch_size=32, shuffle=True)
    loaderValidacao = DataLoader(datasetValidacao, batch_size=32, shuffle=False)'''     #tive que aumentar o batch_size (fazer rodar mais ampstras de uma só vez) para saturar a GPU para tentar ter mais iterações por segundo diminuindo o número das iterações também --> a rede treina com 256 janelas da trajetória ao mesmo tempo e só atualiza a LOSS após terminar tais janelas
    loaderTreino = DataLoader(datasetTreino, batch_size=256, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)     #tira o preparo dos batches do processo principal (python vira paralelo via workers) e usa a memória pinada para acelerar a transferência da CPU para a GPU (um gargalo existente)
    loaderValidacao = DataLoader(datasetValidacao, batch_size=256, shuffle=False)

    # ------------------- Diagnóstico ANTES de treinar -----------------
    lossPersistenciaRef = baselineDePersistencia(loaderValidacao)
    print(f"Baseline de persistência (val, sem rede nenhuma): {lossPersistenciaRef:.6f}")

    # ------------------- Modelo ----------------------------------------
    modelo = RedeTransformer(dModel=D_MODEL, nHeads=N_HEADS, nCamadas=N_CAMADAS,
                              dimFeedforward=DIM_FEEDFORWARD, dropout=DROPOUT).to(device)
    otimizador = t.optim.Adam(modelo.parameters(), lr=1e-3)     #ATUALIZA OS PARÂMETROS BASEADO NOS GRADIENTES CALCULADOS PELO .backward()

    #vou adicionar um scheduler de redução de learning rate de acordo com a função COSSENO, não reativo aos platôs, mas sim uma redução pequena no inicio e no fim e grande no meio
    scheduler=t.optim.lr_scheduler.CossineAnnealignLR(otimizador,T_max=N_EPOCAS)

    ultimoCheckpoint,epocaInicial=acharUltimoCheckpoint(".")
    if ultimoCheckpoint is not None:
        print(f"Retomando checkpoint: {ultimoCheckpoint.name} (época {epocaInicial})")
        modelo.load_state_dict(t.load(ultimoCheckpoint, map_location=device))
    else:
        print("Nenhum checkpoint encontrado --> treinando do zero a rede")
    # ------------------- Loop de treino ---------------------------------
    historicoLossVal = []
    for epoca in range(epocaInicial, N_EPOCAS + 1):
        modelo.train()
        lossTreinoAcumulada = 0.0
        #CÁLCULO DA PROBABILIDADE DE TEACHER FORCING DESTA ÉPOCA EM ESPECÍFICO
        probTF=probabilidadeTeacherForcing(epoca-1)
    
        for janela, massa, futuros in tqdm(loaderTreino, desc=f"Época {epoca}"):
        #for janela, massa, futuros in loaderTreino:
            janela, massa, futuros = janela.to(device), massa.to(device), futuros.to(device)
            otimizador.zero_grad() #AQUI OS GRADIENTES CALCULADOS NO BATCH ANTERIOR NÃO PASSAM PARA O PRÓXIMO BATCH

            B=janela.shape[0]
            janelaAtual = janela
            lossPassos = []
            for h in range(HORIZONTE_MAX):
                alvo = futuros[:, h, :]
                ultimoEstado = janelaAtual[:, -1, :]  #usado para o cálculo da loss de consistência
                previsto = modelo(janelaAtual, massa)

                loss, termos = lossTotal(desnormalizar(previsto,media,desvio), desnormalizar(alvo,media,desvio), desnormalizar(ultimoEstado,media,desvio), desnormalizar(massa,mediaMassa,desvioMassa), DT, PESOS_LOSS)
                lossPassos.append(loss)

                # ================================================================ SHEDULED SAMPLING ================================================================
                #aqui eu decido para cada amostra do batch se irei usar o valor real -teacher forcing- ou o previsão da própria rede
                usarReal = t.rand(B, device=device) < probTF   #é basicamente decidir quais batches terão que usar a previsão da rede: gera B numeros aleatorios entre 0 e 1 para cada amostra do batch, então retorna, por exemplo se B=4 [0.83, 0.1, 0.57, 0.3], e se probTF=0.4 o usarReal será [False, True, False, True] --> (B,)
                usarReal = usarReal.unsqueeze(1)               
                #(B,1) para poder bater as dimensões, então vai de
                #[False, True, False, True] --> shape (4,) para (4,1)
                #[[False],
                # [True], 
                # [False],
                # [True]] 

                #aqui tenho .where(condição,X,Y), onde, se condição é True pego o valor de X, e se é False pego o valor de Y
                #.detach() foi ideia de IA pois corta o grafo de histórico do tensor, evitando o BACKPROPAGATION THROUGH TIME, que seria acumular grafo entre os passos e podendo alterar a LOSS de uma maneira ruim e inconsistente (poderia explodir)
                proximoEstado = t.where(usarReal, alvo, previsto.detach())

                #proximoEstado = alvo   #teacher forcing puro

                janelaAtual = t.cat([janelaAtual[:, 1:, :], proximoEstado.unsqueeze(1)], dim=1) #desliza a janela (janelaAtual[:, 1:, :] pega os passos da janelaAtual a partir do segundo índice) e adiciona/concatena o proximo estado

            #lossPassos é uma lista que foi preenchida pela previsao da rede após treinar com a janela dada
            lossFinal = t.stack(lossPassos).mean()  #faz com que cada tensor de lossPassos seja um elemento de um tensor de 1 dimensão, e o .mean() traz a média desses tensores todos, a média entre todos os passos do horizonte definido (definiria o quão bem o moedlo doi em média prevendo HORIZONTE_MAX)
            '''lossFinal.backward()                    #backpropagation --> irá percorrer o grafo computacional do PyTorch de trás pra frente calculando o gradiente da LOSS de acordo com cada parâmetro treinável do modelo, que ficará guardado em .grad de cada parâmetro --> vai definir para qual direçãõ vai ser alterado o parâmetro
			t.nn.utils.clip_grad_norm_(modelo.parameters(),max_norm=1.0)
            otimizador.step()'''                       #onde há o aprendizado chamando o otimizador para cada parâmetro
            lossFinal.backward()
            t.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            lossTreinoAcumulada += lossFinal.item() #o .item() retira o valor float32 do tensor do torch, sem graafo que se acumula
        t.save(modelo.state_dict(), f"checkpointEpoca{epoca:03d}.pt")
        lossTreinoAcumulada /= len(loaderTreino)
        print(f"Época {epoca:3d} | loss do treino: {lossTreinoAcumulada:.6f}")
        scheduler.step()
    
    t.save({
        "state_dict": modelo.state_dict(),
        "media": media, "desvio": desvio,
        "mediaMassa": mediaMassa, "desvioMassa": desvioMassa,
    }, "modelo3corpos.pt")
    print("Modelo salvo em modelo3corpos.pt")
    
    print("\n--- Rollout autoregressivo ---")
    horizontes, errosMedios = avaliarRollout(modelo,trajetoriasValidacao,massasValidacao,TAMANHO_JANELA,media,desvio,mediaMassa,desvioMassa,device,nTrajetoriasMax=12, nPassosDoRollout=300)
