import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotly import graph_objects as go
from multiprocessing import Pool
import os

def extrair_horizons(arquivoEntrada):
    dados={
        'X':[],
        'Y':[],
        'Z':[],
        'VX':[],
        'VY':[],
        'VZ':[]
    }    #dicionario
    permitidos=['X','Y','Z','VX','VY','VZ']
    with open(arquivoEntrada,"r",encoding="utf-8") as arquivo: #o with open fecha o arquivo logo após ser usado
        conteudo=arquivo.read().split()
        #print(conteudo)
        i=0
        while i<len(conteudo):
            #TRATAMENTO DE 'VX' E 'VY'
            if '=' in conteudo[i]:
                chave,valor=conteudo[i].split('=',1)
                if chave in permitidos:
                    if valor!="":
                        try:
                            dados[chave].append(float(valor))
                        except ValueError:
                            pass
                    else:
                        j=i+1
                        while j<len(conteudo) and conteudo[j]=='':
                            j+=1
                        if j< len(conteudo):
                            try:
                                dados[chave].append(float(conteudo[j]))
                                i=j
                            except ValueError:
                                pass
            #TRATAMENTO DE 'X' E 'Y'
            elif conteudo[i] in permitidos and i+2<len(conteudo) and conteudo[i+1]=='=':
                try:
                    dados[conteudo[i]].append(float(conteudo[i+2]))
                    i+=2
                except ValueError:
                    pass
            elif i+1 < len(conteudo) and conteudo[i+1].startswith("=") and conteudo[i] in permitidos:
                valor=conteudo[i+1].replace("=","")
                if valor!='':
                    try:
                       dados[conteudo[i]].append(float(valor))
                    except ValueError:
                        pass
            i+=1

    return dados

def salvarDadosHorizonsNPZ(dados,nome):
    X=np.array(dados["X"])
    Y=np.array(dados["Y"])
    Z=np.array(dados["Z"])
    VX=np.array(dados["VX"])
    VY=np.array(dados["VY"])
    VZ=np.array(dados["VZ"])
   
    np.savez_compressed(f"simulacoesArtificiais/horizons{nome}.npz",X=X,Y=Y,Z=Z,VX=VX,VY=VY,VZ=VZ)
    print(f"Arquivo horizons{nome} salvo. Tamanho aproximado: {(X.nbytes+Y.nbytes+Z.nbytes+VX.nbytes+VY.nbytes+VZ.nbytes)/1024:.2f} KB")

#função para carregar os dados depois
def carregarDadosHorizonsNPZ(nome):
    dados=np.load(f"simulacoesArtificiais/horizons{nome}.npz")
    return (dados["X"],dados["Y"],dados["Z"],dados["VX"],dados["VY"],dados["VZ"])

def salvarEstadosNPZ(massas,trajetoria,tempo,energia,momAng,momLin,rMomentaneo,aceleracoes,nome,dt):
    #convertendo para array caso ainda não seja
    massas=np.array(massas)
    trajetoria=np.array(trajetoria)
    tempo=np.array(tempo)
    energia=np.array(energia)
    momAng=np.array(momAng)
    momLin=np.array(momLin)
    rMomentaneo=np.array(rMomentaneo)
    aceleracoes=np.array(aceleracoes)


    #para facilitar no treinamento da rede, salvo todos os arrays em um arquivo compactado
    np.savez_compressed(f"simulacoesArtificiais/simulacao3D{nome}.npz",
                        massas=massas,trajetoria=trajetoria,tempo=tempo,energia=energia,momAng=momAng,momLin=momLin,rMomentaneo=rMomentaneo,aceleracoes=aceleracoes,dt=dt)
    print(f"Arquivo salvo. Tamanho aproximado: {trajetoria.nbytes/1e6:.1f} MB")

#função para carregar os dados depois
def carregarEstadosNPZ(nome):
    dados=np.load(f"simulacoesArtificiais2C/simulacoesTeste/simulacao3D_estruturaTeste2C{nome}.npz")
    return [dados['estado'],dados['massas']]

'''def yoshida4ordem(estado,dt,m1,m2,m3):#utiliza algumas vezes o velocity-verlet
    w1=1/(2-2**(1/3))
    w0=-2**(1/3)/(2-2**(1/3))
    c1=c4=w1/2
    c2=c3=(w0+w1)/2
    d1=d3=w1
    d2=w0
    estado=estado.copy()

    r1=estado[0:3]
    v1=estado[3:6]
    r2=estado[6:9]
    v2=estado[9:12]
    r3=estado[12:15]
    v3=estado[15:18]

    # PRIMEIRA PARTE
    #DRIFT
    r1+=c1*dt*v1
    r2+=c1*dt*v2
    r3+=c1*dt*v3

    a1,a2,a3=atualizaAceleracoes_posicoes(r1,r2,r3,m1,m2,m3)
    #KICK
    v1+=d1*dt*a1
    v2+=d1*dt*a2
    v3+=d1*dt*a3

    # SEGUNDA PARTE
    #DRIFT
    r1+=c2*dt*v1
    r2+=c2*dt*v2
    r3+=c2*dt*v3

    a1,a2,a3=atualizaAceleracoes_posicoes(r1,r2,r3,m1,m2,m3)
    #KICK
    v1+=d2*dt*a1
    v2+=d2*dt*a2
    v3+=d2*dt*a3

    # TERCEIRA PARTE
    #DRIFT
    r1+=c3*dt*v1
    r2+=c3*dt*v2
    r3+=c3*dt*v3

    a1,a2,a3=atualizaAceleracoes_posicoes(r1,r2,r3,m1,m2,m3)
    #KICK
    v1+=d3*dt*a1
    v2+=d3*dt*a2
    v3+=d3*dt*a3

    # QUARTA PARTE
    #DRIFT
    r1+=c4*dt*v1
    r2+=c4*dt*v2
    r3+=c4*dt*v3

    return np.concatenate([r1,v1,r2,v2,r3,v3])'''
def yoshida4ordem(estado,dt,m1,m2):#utiliza algumas vezes o velocity-verlet
    w1=1/(2-2**(1/3))
    w0=-2**(1/3)/(2-2**(1/3))
    c1=c4=w1/2
    c2=c3=(w0+w1)/2
    d1=d3=w1
    d2=w0
    estado=estado.copy()

    r1=estado[0:3]
    v1=estado[3:6]
    r2=estado[6:9]
    v2=estado[9:12]


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

'''def atualizaAceleracoes_posicoes(r1,r2,r3,m1,m2,m3):
    #vetor posição
    pr12=r2-r1
    pr13=r3-r1
    pr23=r3-r2
    #radicando
    rr12=pr12[0]**2 + pr12[1]**2 + pr12[2]**2 + epsilon**2
    rr13=pr13[0]**2 + pr13[1]**2 + pr13[2]**2 + epsilon**2
    rr23=pr23[0]**2 + pr23[1]**2 + pr23[2]**2 + epsilon**2
    #fazendo o inverso para poupar algumas divisões
    inv_r12=1/(rr12*np.sqrt(rr12))
    inv_r13=1/(rr13*np.sqrt(rr13))
    inv_r23=1/(rr23*np.sqrt(rr23))

    a1=G*(m2*pr12*inv_r12 + m3*pr13*inv_r13)
    a2=G*(-m1*pr12*inv_r12 + m3*pr23*inv_r23)
    a3=G*(-m1*pr13*inv_r13 - m2*pr23*inv_r23)

    return a1,a2,a3'''
def atualizaAceleracoes_posicoes(r1,r2,m1,m2):
    #vetor posição
    pr12=r2-r1

    #radicando
    rr12=pr12[0]**2 + pr12[1]**2 + pr12[2]**2 + epsilon**2

    #fazendo o inverso para poupar algumas divisões
    inv_r12=1/(rr12*np.sqrt(rr12))
    
    a1=G*(m2*pr12*inv_r12)
    a2=G*(-m1*pr12*inv_r12)

    return a1,a2

'''def calculaEnergiaDoSistema(estado,m1,m2,m3): #e momentos linear e angular
    x1,y1,z1,vx1,vy1,vz1,\
    x2,y2,z2,vx2,vy2,vz2,\
    x3,y3,z3,vx3,vy3,vz3=estado[:18] #tira o tempo do estado para que não tena uma variável inútil sendo colocada aqui

    r1=np.array([x1,y1,z1])
    r2=np.array([x2,y2,z2])
    r3=np.array([x3,y3,z3])

    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)
    r13=np.sqrt((np.linalg.norm(r3-r1))**2 + epsilon**2)
    r23=np.sqrt((np.linalg.norm(r3-r2))**2 + epsilon**2)

    cinetica=((vx1**2+vy1**2+vz1**2)*m1/2+(vx2**2+vy2**2+vz2**2)*m2/2+(vx3**2+vy3**2+vz3**2)*m3/2)
    
    potencial=-G*(m1*m2/r12+m1*m3/r13+m2*m3/r23)
    
    return cinetica+potencial

#retorna um valor aleatório de um parâmetro (multiplica por algum valor entre 0 e 1), podendo ser negativo ou positivo
def rd(a):
    return float(a)*(2*np.random.rand()-1)

def calculaMomLin(m1,m2,m3,v1,v2,v3):
    return np.linalg.norm(m1*v1 + m2*v2 + m3*v3)

def calculaMomAng(m1,m2,m3,r1,r2,r3,v1,v2,v3):
    return m1*np.cross(r1,v1)+m2*np.cross(r2,v2)+m3*np.cross(r3,v3)
    #return m1*(r1[0]*v1[1] - r1[1]*v1[0]) + m2*(r2[0]*v2[1] - r2[1]*v2[0]) + m3*(r3[0]*v3[1] - r3[1]*v3[0])'''

def calculaEnergiaDoSistema(estado,m1,m2): #e momentos linear e angular
    x1,y1,z1,vx1,vy1,vz1,\
    x2,y2,z2,vx2,vy2,vz2=estado[:12] #tira o tempo do estado para que não tena uma variável inútil sendo colocada aqui

    r1=np.array([x1,y1,z1])
    r2=np.array([x2,y2,z2])

    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)
    
    cinetica=((vx1**2+vy1**2+vz1**2)*m1/2+(vx2**2+vy2**2+vz2**2)*m2/2)
    
    potencial=-G*(m1*m2/r12)
    
    return cinetica+potencial

#retorna um valor aleatório de um parâmetro (multiplica por algum valor entre 0 e 1), podendo ser negativo ou positivo
def rd(a):
    return float(a)*(2*np.random.rand()-1)

def calculaMomLin(m1,m2,v1,v2):
    return np.linalg.norm(m1*v1 + m2*v2)

def calculaMomAng(m1,m2,r1,r2,v1,v2):
    return m1*np.cross(r1,v1)+m2*np.cross(r2,v2)
    #return m1*(r1[0]*v1[1] - r1[1]*v1[0]) + m2*(r2[0]*v2[1] - r2[1]*v2[0]) + m3*(r3[0]*v3[1] - r3[1]*v3[0])

'''def validarSimulador(trajetoria,massas,dt,calcularEnergia,calcularMomAng,calcularMomLin,integrador):
    massas=np.array(massas)
    trajetoria=np.array(trajetoria)

    #======calculo das invariantes do sistema======
    energias,momLin,momAng=[],[],[]
    for estado in trajetoria[:, :-1]: #aqui o [:, :-1] significa que vai pegar todas as linhas individualmente mas MENOS UMA coluna, que é a final (onde há o tempo, inútil nestes cálculos daqui)
        r1=estado[0:3]
        v1=estado[3:6]
        r2=estado[6:9]
        v2=estado[9:12]
        r3=estado[12:15]
        v3=estado[15:18]
        energias.append(calcularEnergia(estado,massas[0],massas[1],massas[2]))
        momLin.append(calcularMomLin(massas[0],massas[1],massas[2],v1,v2,v3))
        momAng.append(calcularMomAng(massas[0],massas[1],massas[2],r1,r2,r3,v1,v2,v3))
    energias=np.array(energias)
    momLin=np.array(momLin)
    momAng=np.array(momAng)

    desvioEnerg=(energias.max()-energias.min())/abs(energias.mean())
    #aqui precisei colocar baseando-se na dimensão dos vetores dos momentos para evitar runtime error na simulação 2D, e como é uma função genérica para os 4 simuladores foi necessário manter como condição mesmo
    escala = np.abs(massas[0] * np.linalg.norm(trajetoria[0, 3:6]))
    #HOUSTON, WE HAVE A PROBLEM...
    if momLin.ndim==1:
        desvioLin=np.max(np.abs(momLin))/escala#np.abs(momLin[0])
    else:
        desvioLin=np.max(np.linalg.norm(momLin,axis=-1))/escala
    if momAng.ndim==1:
        desvioAng=np.max(np.abs(momAng-momAng[0]))/abs(momAng[0])
    else:
        desvioAng=np.max(np.linalg.norm(momAng-momAng[0],axis=-1))/np.linalg.norm(momAng[0]) #lembrando que np.linalg.norm retorna a magnitude, e nesse caso é a magnitude da diferença entre momAng inicial e o atual

    #======reversibilidade======
    estado=trajetoria[0, :-1].copy() #copia a primeira linha sem o tempo
    steps=50000#len(trajetoria)
    for _ in range(steps):
        estado=integrador(estado,dt,massas[0],massas[1],massas[2])
    for _ in range(steps):
        estado=integrador(estado,-dt,massas[0],massas[1],massas[2])
    erroReversao=np.linalg.norm(estado-trajetoria[0, :-1])/np.linalg.norm(trajetoria[0, :-1])

    return [desvioEnerg,desvioLin,desvioAng,erroReversao]'''
def validarSimulador(trajetoria,massas,dt,calcularEnergia,calcularMomAng,calcularMomLin,integrador):
    massas=np.array(massas)
    trajetoria=np.array(trajetoria)

    #======calculo das invariantes do sistema======
    energias,momLin,momAng=[],[],[]
    for estado in trajetoria[:, :-1]: #aqui o [:, :-1] significa que vai pegar todas as linhas individualmente mas MENOS UMA coluna, que é a final (onde há o tempo, inútil nestes cálculos daqui)
        r1=estado[0:3]
        v1=estado[3:6]
        r2=estado[6:9]
        v2=estado[9:12]

        energias.append(calcularEnergia(estado,massas[0],massas[1]))
        momLin.append(calcularMomLin(massas[0],massas[1],v1,v2))
        momAng.append(calcularMomAng(massas[0],massas[1],r1,r2,v1,v2))
    energias=np.array(energias)
    momLin=np.array(momLin)
    momAng=np.array(momAng)

    desvioEnerg=(energias.max()-energias.min())/abs(energias.mean())
    #aqui precisei colocar baseando-se na dimensão dos vetores dos momentos para evitar runtime error na simulação 2D, e como é uma função genérica para os 4 simuladores foi necessário manter como condição mesmo
    escala = np.abs(massas[0] * np.linalg.norm(trajetoria[0, 3:6]))
    #HOUSTON, WE HAVE A PROBLEM...
    if momLin.ndim==1:
        desvioLin=np.max(np.abs(momLin))/escala#np.abs(momLin[0])
    else:
        desvioLin=np.max(np.linalg.norm(momLin,axis=-1))/escala
    if momAng.ndim==1:
        desvioAng=np.max(np.abs(momAng-momAng[0]))/abs(momAng[0])
    else:
        desvioAng=np.max(np.linalg.norm(momAng-momAng[0],axis=-1))/np.linalg.norm(momAng[0]) #lembrando que np.linalg.norm retorna a magnitude, e nesse caso é a magnitude da diferença entre momAng inicial e o atual

    #======reversibilidade======
    estado=trajetoria[0, :-1].copy() #copia a primeira linha sem o tempo
    steps=50000#len(trajetoria)
    for _ in range(steps):
        estado=integrador(estado,dt,massas[0],massas[1])
    for _ in range(steps):
        estado=integrador(estado,-dt,massas[0],massas[1])
    erroReversao=np.linalg.norm(estado-trajetoria[0, :-1])/np.linalg.norm(trajetoria[0, :-1])

    return [desvioEnerg,desvioLin,desvioAng,erroReversao]




#============================== VARIÁVEIS DA SIMULAÇÃO =================================


#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsLua60d.txt'),"LUA60d")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsTerra60d.txt'),"TERRA60d")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsSol60d.txt'),"SOL60d")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsSol1a6h.txt'),"SOL1a6h")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsLua1a6h.txt'),"LUA1a6h")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsTerra1a6h.txt'),"TERRA1a6h")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsPlutao.txt'),"PLUTAO")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsCaronte.txt'),"CARONTE")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsMarte.txt'),"MARTE")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsFobos.txt'),"FOBOS")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsJupiter.txt'),"JUPITER")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsIo.txt'),"IO")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsEuropa.txt'),"EUROPA")

estado0=np.array([
    -10, -5, 6, .11, .5, .4,
     10, -10, 3, -.2, -.6, .8,
],dtype=float)

mTerra=5.97219e24
mLua=7.349e22
mSol=1988410e24
mPlutao=1.307e22
mCaronte=1.586e21
mMarte=6.4171e23
mFobos=1.08e16
mJupiter=18.9819e26
mIo=8.93e22
mEuropa=4.79984e22

trajetoria=[]
tempoSimulacao=[]
energiaDoSistema=[]
energiaHorizons=[]
diferencaDistanciaHorizonsXSimulacao=[]
r_dinamicoSistema=[]
r_dinamicoHorizons=[]
momLinSistema=[]
momAngSistema=[]
momLinHorizons=[]
momAngHorizons=[]
aceleracoes=[]

diferencaEnergiaHorizonsXSimulacao=[]
diferencaMomLinHorizonsXSimulacao=[]
diferencaMomAngHorizonsXSimulacao=[]
erro_posicao=[]
erro_relativo_posicao=[]

steps=400000
N=0#VALORES PADRÃO
dt=0.00025#VALORES PADRÃO
G=1
epsilon=1e-6  #0  #1e-2
num_simul=1
j=0

#flag = 0  -->  simulador ALEATÓRIO
#flag = 1  -->  simulador TERRA-LUA
#flag = 2  -->  simulador PLUTÃO-CARONTE
#flag = 3  -->  simulador MARTE-PHOBOS
#flag = 4  -->  simulador JÚPITER-IO
#flag = 5  -->  verificar simulação gerada pelo DATASET

flagTipoDeSimulacao=5



#============================== SIMULAÇÃO =================================

if flagTipoDeSimulacao==0:
    while j<num_simul:
        rMomentaneo=[]
        momLin=[]
        momAng=[]
        aceleracoes=[]

        x1,y1,z1,vx1,vy1,vz1,\
        x2,y2,z2,vx2,vy2,vz2=estado0

        #alternado as massas (mas mantendo a massa total = 1)
        m1simulacao=np.random.rand()
        m2simulacao=1-m1simulacao

        #alterando as posições e velocidades iniciais do sistema
        estadoAleatorio=np.array([rd(a) for a in[ #gera numeros aleatórios baseando-se nos valores origianais do estado0, sem alterá-los no molde
            x1,y1,z1,vx1,vy1,vz1,\
            x2,y2,z2,vx2,vy2,vz2
        ]])
        '''print("ESTADO0: ",estado0)
        print("MASSA ORIGINAL: ",m1simulacao,m2simulacao)
        print("\nESTADO GERADO: ",estado)
        print("\nMASSA ALTERADA: ",m1simulacao,m2simulacao,"\n")
        print("SOMA MASSAS:",m1simulacao+m2simulacao)'''
        x1,y1,z1,vx1,vy1,vz1,x2,y2,z2,vx2,vy2,vz2=estadoAleatorio

        #mantendo o momento linear do sistema, forçadamente, em ZERO        P_total=m1simulacao*v1+m2simulacao*v2+m3simulacao*v3       
        r1=np.array([x1,y1,z1])
        r2=np.array([x2,y2,z2])
        v1=np.array([vx1,vy1,vz1])
        v2=np.array([vx2,vy2,vz2])

        V_cm=(m1simulacao*v1+m2simulacao*v2)/(m1simulacao+m2simulacao)

        v1-=V_cm
        v2-=V_cm


        #colocando o centro de massa do sistema na origem da simulação, para facilitar o treinamento da rede (evita que ela tenha que aprender o DRIFT - o centro de massa estar em certa posição não afeta na evolução da simulação)
        R_cm=(m1simulacao*r1+m2simulacao*r2)/(m1simulacao+m2simulacao)

        r1-=R_cm
        r2-=R_cm

        estado = np.array([
            r1[0],r1[1],r1[2],v1[0],v1[1],v1[2],
            r2[0],r2[1],r2[2],v2[0],v2[1],v2[2]
        ])

        trajetoria=[]
        tempoSimulacao=[]
        energiaDoSistema=[]
        
        passoAtual=0
        flag_colisao=0
        tAtual=0.0

        for i in range(steps):
            passoAtual+=1
            estadoComTempo=np.append(estado.copy(),tAtual)
            trajetoria.append(estadoComTempo)#aqui o trajetoria é uma lista de arrays
            tempoSimulacao.append(tAtual)

            energiaDoSistema.append(calculaEnergiaDoSistema(estado,m1simulacao,m2simulacao))

            #PARA EVITAR COLISÕES, QUE NÃO É O OBJETIVO DA REDE COMPREENDER, COLOCO ESSE GATILHO PARA EVITAR COLOCAR OS DADOS DA TRAJETÓRIA NO DATASET
            r12=np.sqrt((np.linalg.norm(np.array([estado[6],estado[7],estado[8]])-np.array([estado[0],estado[1],estado[2]])))**2 + epsilon**2)

            r1=estado[0:3]
            v1=estado[3:6]
            r2=estado[6:9]
            v2=estado[9:12]

            a1,a2=atualizaAceleracoes_posicoes(r1,r2,m1simulacao,m2simulacao)
            aceleracoes.append(np.concatenate([a1,a2]))

            #evolução do sistema
            estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao)

            momLin.append(calculaMomLin(m1simulacao,m2simulacao,v1,v2))
            momAng.append(calculaMomAng(m1simulacao,m2simulacao,r1,r2,v1,v2))

            #a distância momentânea nao se aplica mais dessa forma, teria que ter soma da distância entre os corpos pelo menos para fazer sentido
            rMomentaneo.append(np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2))

            #progresso da simulação em porcentagem 
            if passoAtual % 10000 == 0 and passoAtual > 0:
                print(f"Passo {passoAtual}/{steps} ({100*passoAtual/steps:.1f}%)")
            if r12<1 or energiaDoSistema[0]>0:
                print("\nCOLISAO DETECTADA ou SISTEMA HIPERBÓLICO, CANCELANDO SIMULAÇÃO")
                print("\nPASSO \n\n",passoAtual)
                flag_colisao=1
                break

            tAtual+=dt
            
        #salva os dados da simulação se não houve colisão
        if(flag_colisao==0):  
            massas=[m1simulacao,m2simulacao]
            salvarEstadosNPZ(massas,trajetoria,tempoSimulacao,energiaDoSistema,momAng,momLin,rMomentaneo,aceleracoes,j,dt)
            j+=1
            trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos
            energiaDoSistema=np.array(energiaDoSistema)

            #==============VALIDANDO O SIMULADOR==============
            resultados=validarSimulador(trajetoria,massas,dt,calculaEnergiaDoSistema,calculaMomAng,calculaMomLin,yoshida4ordem)
            print(f"VALORES DA VALIDAÇÃO:\nDesvio da energia: {resultados[0]}\nDesvio do momento linear: {resultados[1]}\nDesvio do momento angular: {resultados[2]}\nErro da reversão: {resultados[3]:.2e}")

            x1=trajetoria[:,0]
            y1=trajetoria[:,1]
            z1=trajetoria[:,2]

            x2=trajetoria[:,6]
            y2=trajetoria[:,7]
            z2=trajetoria[:,8]


            fig, axs = plt.subplots(2, 2, figsize=(12, 12))

            # ---------------- TRAJETÓRIA 3D ----------------
            # número de frames da animação
            n_pontos = 500
            step_anim = max(1, len(x1) // n_pontos)
            x1a, y1a, z1a = x1[::step_anim], y1[::step_anim], z1[::step_anim]
            x2a, y2a, z2a = x2[::step_anim], y2[::step_anim], z2[::step_anim]

            n_frames = 100  # menos frames também ajuda
            indices = np.linspace(0, len(x1a)-1, n_frames, dtype=int)

            frames = []
            for k, idx in enumerate(indices):
                frames.append(go.Frame(
                    data=[
                        go.Scatter3d(x=x1a[:idx+1], y=y1a[:idx+1], z=z1a[:idx+1],
                                    mode='lines', line=dict(color='blue', width=2), name='corpo 1'),
                        go.Scatter3d(x=x2a[:idx+1], y=y2a[:idx+1], z=z2a[:idx+1],
                                    mode='lines', line=dict(color='red', width=2), name='corpo 2'),
                        go.Scatter3d(x=[x1a[idx]], y=[y1a[idx]], z=[z1a[idx]],
                                    mode='markers', marker=dict(size=6, color='blue'), showlegend=False),
                        go.Scatter3d(x=[x2a[idx]], y=[y2a[idx]], z=[z2a[idx]],
                                    mode='markers', marker=dict(size=6, color='red'), showlegend=False),
                    ],
                    name=str(k)
                ))
            

            # estado inicial
            figTrajetoria = go.Figure(
                data=[
                    go.Scatter3d(x=[x1a[0]], y=[y1a[0]], z=[z1a[0]],
                                mode='lines+markers', line=dict(color='blue'), name='corpo 1'),
                    go.Scatter3d(x=[x2a[0]], y=[y2a[0]], z=[z2a[0]],
                                mode='lines+markers', line=dict(color='red'), name='corpo 2'),
                    go.Scatter3d(x=[x1a[0]], y=[y1a[0]], z=[z1a[0]],
                                mode='markers', marker=dict(size=6, color='blue'), showlegend=False),
                    go.Scatter3d(x=[x2a[0]], y=[y2a[0]], z=[z2a[0]],
                                mode='markers', marker=dict(size=6, color='red'), showlegend=False),
                ],
                frames=frames
            )

            duracao_ms = 10000  # 10 segundos
            ms_por_frame = duracao_ms // n_frames

            figTrajetoria.update_layout(
                title="Simulação 3D - 2 Corpos",
                updatemenus=[dict(
                    type='buttons',
                    showactive=False,
                    y=1.05, x=0.1,
                    buttons=[
                        dict(label='▶ Play',
                            method='animate',
                            args=[None, dict(frame=dict(duration=ms_por_frame, redraw=True),
                                            fromcurrent=True, mode='immediate')]),
                        dict(label='⏸ Pause',
                            method='animate',
                            args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                mode='immediate')])
                    ]
                )],
                sliders=[dict(
                    steps=[dict(method='animate',
                                args=[[str(k)], dict(mode='immediate',
                                                    frame=dict(duration=ms_por_frame, redraw=True))],
                                label=str(k)) for k in range(n_frames)],
                    x=0.1, y=0, len=0.9
                )]
            )

            figTrajetoria.show()
            '''figTrajetoria = go.Figure()
            figTrajetoria.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='lines', name='corpo 1'))
            figTrajetoria.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='lines', name='corpo 2'))
            figTrajetoria.add_trace(go.Scatter3d(x=x3, y=y3, z=z3, mode='lines', name='corpo 3'))
            figTrajetoria.add_trace(go.Scatter3d(x=[x1[0]], y=[y1[0]], z=[z1[0]], mode='markers', showlegend=False))
            figTrajetoria.add_trace(go.Scatter3d(x=[x2[0]], y=[y2[0]], z=[z2[0]], mode='markers', showlegend=False))
            figTrajetoria.add_trace(go.Scatter3d(x=[x3[0]], y=[y3[0]], z=[z3[0]], mode='markers', showlegend=False))
            figTrajetoria.update_layout(title="Trajetória 3D")
            figTrajetoria.show()'''
            
            # ---------------- TRAJETÓRIA ----------------
            axs[0,0].plot(x1, y1, label='corpo 1')
            axs[0,0].plot(x2, y2, label='corpo 2')
            axs[0,0].scatter(x1[0], y1[0])
            axs[0,0].scatter(x2[0], y2[0])
            axs[0,0].set_title("Trajetória")
            axs[0,0].axis("equal")
            axs[0,0].grid()

            # ---------------- Erro relativo da posição em porcentagem ----------------
            axs[0,1].plot(erro_relativo_posicao)
            #erro_absoluto_km = np.array(erro_posicao) / 1e3
            #axs[0,1].plot(erro_absoluto_km)
            axs[0,1].set_title("Erro relativo da posição em porcentagem")
            axs[0,1].grid()

            # ---------------- Energia ----------------
            axs[1,1].plot(energiaDoSistema)
            axs[1,1].set_title("Energia Total")
            axs[1,1].ticklabel_format(useOffset=False, style='sci', axis='y')
            axs[1,1].grid()

            # ---------------- Erro relativo SIMULADOR----------------
            E0 = energiaDoSistema[0]
            print("Órbita ligada" if E0<0 else "Hiperbólica - não ligado")
            erro_relativo = (energiaDoSistema - E0)/abs(E0)
            axs[1,0].plot(erro_relativo)
            axs[1,0].set_title("Erro Relativo da Energia do SIMULADOR")
            axs[1,0].grid()

            plt.tight_layout()
            plt.show()


else: 
    G=6.6743e-11
    epsilon=1e-12
    if flagTipoDeSimulacao==1:
        '''N=120 #numero de subpassos
        dt=3600/N'''
        #steps = 721     #30 dias 1h
        #steps = 1440     #60 dias 1h
        #steps = 1460    #365 dias 6h
        N = 720               # subpassos
        dt = 6*3600 / N
        
        m1simulacao=mTerra
        m2simulacao=mLua
        #m3simulacao=mSol
        #xs,ys,zs,vxs,vys,vzs=carregarDadosHorizonsNPZ("SOL1a6h")
        xt,yt,zt,vxt,vyt,vzt=carregarDadosHorizonsNPZ("TERRA1a6h")
        xl,yl,zl,vxl,vyl,vzl=carregarDadosHorizonsNPZ("LUA1a6h")

        '''xs,ys,zs,vxs,vys,vzs=carregarDadosHorizonsNPZ("SOL60d")
        xt,yt,zt,vxt,vyt,vzt=carregarDadosHorizonsNPZ("TERRA60d")
        xl,yl,zl,vxl,vyl,vzl=carregarDadosHorizonsNPZ("LUA60d")'''

        '''xs,ys,zs,vxs,vys,vzs=carregarDadosHorizonsNPZ("SOL")
        xt,yt,zt,vxt,vyt,vzt=carregarDadosHorizonsNPZ("TERRA")
        xl,yl,zl,vxl,vyl,vzl=carregarDadosHorizonsNPZ("LUA")'''

        steps = len(xt)
    
    elif flagTipoDeSimulacao==2:
        steps = 721
        N=12
        dt=3600/N

        m1simulacao=mPlutao
        m2simulacao=mCaronte
        #m3simulacao=mSol
        #xs,ys,zs,vxs,vys,vzs=carregarDadosHorizonsNPZ("SOL")
        xt,yt,zt,vxt,vyt,vzt=carregarDadosHorizonsNPZ("PLUTAO")
        xl,yl,zl,vxl,vyl,vzl=carregarDadosHorizonsNPZ("CARONTE")   

    elif flagTipoDeSimulacao==3:
        N=120 #passos entre uma verificação e outra - assim vai rodar mais passos sem ter que comparar os dados da HORIZONS
        dt=600/N
        steps = 721
    
        m1simulacao=mMarte
        m2simulacao=mFobos
        #m3simulacao=mSol
        #xs,ys,zs,vxs,vys,vzs=carregarDadosHorizonsNPZ("SOL")
        xt,yt,zt,vxt,vyt,vzt=carregarDadosHorizonsNPZ("MARTE")
        xl,yl,zl,vxl,vyl,vzl=carregarDadosHorizonsNPZ("FOBOS")
    
    elif flagTipoDeSimulacao==4:
        N=120 #passos entre uma verificação e outra - assim vai rodar mais passos sem ter que comparar os dados da HORIZONS
        dt=3600/N
        #steps = 721
        
        m1simulacao=mJupiter
        m2simulacao=mIo
        #m3simulacao=mEuropa
        #xs,ys,zs,vxs,vys,vzs=carregarDadosHorizonsNPZ("EUROPA")
        xt,yt,zt,vxt,vyt,vzt=carregarDadosHorizonsNPZ("JUPITER")
        xl,yl,zl,vxl,vyl,vzl=carregarDadosHorizonsNPZ("IO")
        
        steps = len(xt)

    elif flagTipoDeSimulacao==5:
        dt=0.00025
        N=40
        G=1
        epsilon=1e-6
        trajetoriaDataSet,massas=carregarEstadosNPZ(104) #NOME DA SIMULAÇÃO A SER TESTADA
        xt=trajetoriaDataSet[:,0]
        yt=trajetoriaDataSet[:,1]
        zt=trajetoriaDataSet[:,2]
        vxt=trajetoriaDataSet[:,3]
        vyt=trajetoriaDataSet[:,4]
        vzt=trajetoriaDataSet[:,5]
        xl=trajetoriaDataSet[:,6]
        yl=trajetoriaDataSet[:,7]
        zl=trajetoriaDataSet[:,8]
        vxl=trajetoriaDataSet[:,9]
        vyl=trajetoriaDataSet[:,10]
        vzl=trajetoriaDataSet[:,11]

        m1simulacao,m2simulacao=massas[0],massas[1]
        estado=trajetoriaDataSet[0, :12]
        x1,y1,z1,vx1,vy1,vz1=xt[0],yt[0],zt[0],vxt[0],vyt[0],vzt[0]
        x2,y2,z2,vx2,vy2,vz2=xl[0],yl[0],zl[0],vxl[0],vyl[0],vzl[0]#isso substituiria a declaração dos valores de X e Y em estadoTerraLua... burrice minha, simplesmente

        steps = len(xt)


    if flagTipoDeSimulacao!=5:
        xt,yt,zt,vxt,vyt,vzt=xt*1e3,yt*1e3,zt*1e3,vxt*1e3,vyt*1e3,vzt*1e3
        xl,yl,zl,vxl,vyl,vzl=xl*1e3,yl*1e3,zl*1e3,vxl*1e3,vyl*1e3,vzl*1e3

        x1,y1,z1,vx1,vy1,vz1=xt[0],yt[0],zt[0],vxt[0],vyt[0],vzt[0]
        x2,y2,z2,vx2,vy2,vz2=xl[0],yl[0],zl[0],vxl[0],vyl[0],vzl[0]#isso substituiria a declaração dos valores de X e Y em estadoTerraLua... burrice minha, simplesmente

        R_cm0=(m1simulacao*np.array([x1,y1,z1])+m2simulacao*np.array([x2,y2,z2]))/(m1simulacao+m2simulacao)
        V_cm0=(m1simulacao*np.array([vx1,vy1,vz1])+m2simulacao*np.array([vx2,vy2,vz2]))/(m1simulacao+m2simulacao)

        estado=np.array([
            x1-R_cm0[0],y1-R_cm0[1],z1-R_cm0[2],vx1-V_cm0[0],vy1-V_cm0[1],vz1-V_cm0[2],
            x2-R_cm0[0],y2-R_cm0[1],z2-R_cm0[2],vx2-V_cm0[0],vy2-V_cm0[1],vz2-V_cm0[2],
        ])

    passoAtual=0
    flag_colisao=0
    tAtual=0.0

    for i in range(steps):
        passoAtual+=1
        estadoComTempo=np.append(estado.copy(),tAtual)
        trajetoria.append(estadoComTempo)
        tempoSimulacao.append(tAtual)
        if flagTipoDeSimulacao==5:
            estadoHorizons=trajetoriaDataSet[i,:12] #faço isso pois os dados do dataset já estão centrados no centro de massa
        else:
            #lembrando que o referencial do HORIZONS é diferente do referencial usado na simulação, pois no horizons usa-se o baricentro do sistema solar
            R_cm_HOR=(m1simulacao*np.array([xt[i],yt[i],zt[i]])+m2simulacao*np.array([xl[i],yl[i],zl[i]]))/(m1simulacao+m2simulacao)
            V_cm_HOR=(m1simulacao*np.array([vxt[i],vyt[i],vzt[i]])+m2simulacao*np.array([vxl[i],vyl[i],vzl[i]]))/(m1simulacao+m2simulacao)
            estadoHorizons=[
                xt[i]-R_cm_HOR[0],yt[i]-R_cm_HOR[1],zt[i]-R_cm_HOR[2],vxt[i]-V_cm_HOR[0],vyt[i]-V_cm_HOR[1],vzt[i]-V_cm_HOR[2],
                xl[i]-R_cm_HOR[0],yl[i]-R_cm_HOR[1],zl[i]-R_cm_HOR[2],vxl[i]-V_cm_HOR[0],vyl[i]-V_cm_HOR[1],vzl[i]-V_cm_HOR[2]
            ]
            
        
        #
        energiaDoSistema.append(calculaEnergiaDoSistema(estado,m1simulacao,m2simulacao))
        energiaHorizons.append(calculaEnergiaDoSistema(estadoHorizons,m1simulacao,m2simulacao))
        diferencaEnergiaHorizonsXSimulacao.append(energiaHorizons[i]-energiaDoSistema[i])

        r1=estado[0:3]
        v1=estado[3:6]
        r2=estado[6:9]
        v2=estado[9:12]

        r_dinamicoSistema.append(np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2))

        momLinSistema.append(calculaMomLin(m1simulacao,m2simulacao,v1,v2))
        momAngSistema.append(calculaMomAng(m1simulacao,m2simulacao,r1,r2,v1,v2))

        #dados do horizons
        x1h,y1h,z1h,vx1h,vy1h,vz1h,x2h,y2h,z2h,vx2h,vy2h,vz2h=estadoHorizons

        r1h=np.array([x1h,y1h,z1h])
        r2h=np.array([x2h,y2h,z2h])
        r_dinamicoHorizons.append(np.sqrt((np.linalg.norm(r2h-r1h))**2 + epsilon**2))
        v1h=np.array([vx1h,vy1h,vz1h])
        v2h=np.array([vx2h,vy2h,vz2h])

        momLinHorizons.append(calculaMomLin(m1simulacao,m2simulacao,v1h,v2h))
        momAngHorizons.append(calculaMomAng(m1simulacao,m2simulacao,r1h,r2h,v1h,v2h))

        #append das diferenças
        diferencaMomLinHorizonsXSimulacao.append(momLinHorizons[i]-momLinSistema[i])
        diferencaMomAngHorizonsXSimulacao.append(momAngHorizons[i]-momAngSistema[i])

        diferencaDistanciaHorizonsXSimulacao.append(r_dinamicoHorizons[i]-r_dinamicoSistema[i])

        #diferença de posição relativa (das somas das posições relativas, na verdade --> inútil, porém interessante)
        r_rel_sim=(estado[6:9]-estado[0:3])  #simulador
        r_rel_hor=np.array([(xl[i]-xt[i]),(yl[i]-yt[i]),(zl[i]-zt[i])]) #HORIZONS

        erro_posicao.append(np.linalg.norm(r_rel_sim-r_rel_hor))

        #evolução do sistema
        for _ in range(N):
            estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao)
        match flagTipoDeSimulacao:
            case 1: tAtual+=3600*6
            case 2: tAtual+=3600
            case 3: tAtual+=600
            case 4: tAtual+=3600
            case 5: tAtual+=N*dt
            case _: tAtual+=1

        #progresso da simulação em porcentagem 
        if passoAtual % 10000 == 0 and passoAtual > 0:
            print(f"Passo {passoAtual}/{steps} ({100*passoAtual/steps:.1f}%)")

    erro_relativo_posicao=100*np.array(erro_posicao)/np.array(r_dinamicoSistema)

    #==============VALIDANDO O SIMULADOR#==============
    massas=[m1simulacao,m2simulacao]
    resultados=validarSimulador(trajetoria,massas,dt,calculaEnergiaDoSistema,calculaMomAng,calculaMomLin,yoshida4ordem)
    print(f"VALORES DA VALIDAÇÃO:\nDesvio da energia: {resultados[0]}\nDesvio do momento linear: {resultados[1]}\nDesvio do momento angular: {resultados[2]}\nErro da reversão: {resultados[3]:.2e}")
    
    #==========================================GERAÇÃO DE PLOTS EM UMA MESMA IMAGEM==========================================
    trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos
    

    #plotando a imagem da trajetória
    x1=trajetoria[:,0]
    y1=trajetoria[:,1]
    z1=trajetoria[:,2]

    x2=trajetoria[:,6]
    y2=trajetoria[:,7]
    z2=trajetoria[:,8]


    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # ---------------- TRAJETÓRIA 3D ----------------
    # número de frames da animação
    n_pontos = 500
    step_anim = max(1, len(x1) // n_pontos)
    x1a, y1a, z1a = x1[::step_anim], y1[::step_anim], z1[::step_anim]
    x2a, y2a, z2a = x2[::step_anim], y2[::step_anim], z2[::step_anim]

    n_frames = 100  # menos frames também ajuda
    indices = np.linspace(0, len(x1a)-1, n_frames, dtype=int)

    frames = []
    for k, idx in enumerate(indices):
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=x1a[:idx+1], y=y1a[:idx+1], z=z1a[:idx+1],
                            mode='lines', line=dict(color='blue', width=2), name='corpo 1'),
                go.Scatter3d(x=x2a[:idx+1], y=y2a[:idx+1], z=z2a[:idx+1],
                            mode='lines', line=dict(color='red', width=2), name='corpo 2'),
                go.Scatter3d(x=[x1a[idx]], y=[y1a[idx]], z=[z1a[idx]],
                            mode='markers', marker=dict(size=6, color='blue'), showlegend=False),
                go.Scatter3d(x=[x2a[idx]], y=[y2a[idx]], z=[z2a[idx]],
                            mode='markers', marker=dict(size=6, color='red'), showlegend=False),
            ],
            name=str(k)
        ))
    

    # estado inicial
    figTrajetoria = go.Figure(
        data=[
            go.Scatter3d(x=[x1a[0]], y=[y1a[0]], z=[z1a[0]],
                        mode='lines+markers', line=dict(color='blue'), name='corpo 1'),
            go.Scatter3d(x=[x2a[0]], y=[y2a[0]], z=[z2a[0]],
                        mode='lines+markers', line=dict(color='red'), name='corpo 2'),
            go.Scatter3d(x=[x1a[0]], y=[y1a[0]], z=[z1a[0]],
                        mode='markers', marker=dict(size=6, color='blue'), showlegend=False),
            go.Scatter3d(x=[x2a[0]], y=[y2a[0]], z=[z2a[0]],
                        mode='markers', marker=dict(size=6, color='red'), showlegend=False),
        ],
        frames=frames
    )

    duracao_ms = 10000  # 10 segundos
    ms_por_frame = duracao_ms // n_frames

    figTrajetoria.update_layout(
        title="Simulação 3D - 2 Corpos",
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=1.05, x=0.1,
            buttons=[
                dict(label='▶ Play',
                    method='animate',
                    args=[None, dict(frame=dict(duration=ms_por_frame, redraw=True),
                                    fromcurrent=True, mode='immediate')]),
                dict(label='⏸ Pause',
                    method='animate',
                    args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate')])
            ]
        )],
        sliders=[dict(
            steps=[dict(method='animate',
                        args=[[str(k)], dict(mode='immediate',
                                            frame=dict(duration=ms_por_frame, redraw=True))],
                        label=str(k)) for k in range(n_frames)],
            x=0.1, y=0, len=0.9
        )]
    )
    figTrajetoria.show()
    
    # ---------------- TRAJETÓRIA ----------------
    axs[0,0].plot(x1, y1, label='corpo 1')
    axs[0,0].plot(x2, y2, label='corpo 2')
    axs[0,0].scatter(x1[0], y1[0])
    axs[0,0].scatter(x2[0], y2[0])
    axs[0,0].set_title("Trajetória")
    axs[0,0].axis("equal")
    axs[0,0].grid()

    # ---------------- Erro relativo da posição em porcentagem ----------------
    axs[0,1].plot(erro_relativo_posicao)
    #erro_absoluto_km = np.array(erro_posicao) / 1e3
    #axs[0,1].plot(erro_absoluto_km)
    axs[0,1].set_title("Erro relativo da posição em porcentagem")
    axs[0,1].grid()

    # ---------------- Energia ----------------
    axs[1,1].plot(energiaDoSistema)
    axs[1,1].set_title("Energia Total")
    axs[1,1].ticklabel_format(useOffset=False, style='sci', axis='y')
    axs[1,1].grid()

    # ---------------- Erro relativo SIMULADOR----------------
    E0 = energiaDoSistema[0]
    print("Órbita ligada" if E0<0 else "Hiperbólica - não ligado")
    erro_relativo = (energiaDoSistema - E0)/abs(E0)
    axs[1,0].plot(erro_relativo)
    axs[1,0].set_title("Erro Relativo da Energia do SIMULADOR")
    axs[1,0].grid()

    plt.tight_layout()
    plt.show()
