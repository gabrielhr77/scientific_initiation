import numpy as np, matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from plotly import graph_objects as go
from multiprocessing import Pool
import os


def salvarEstadosNPZ(massas,estado,tempo,energia,momAng,momLin,nome,dt):
    #convertendo para array caso ainda não seja
    massas=np.array(massas)
    #trajetoria1=np.array(trajetoria1)
    #trajetoria2=np.array(trajetoria2)
    #trajetoria3=np.array(trajetoria3)
    estado=estado
    tempo=np.array(tempo)
    energia=np.array(energia)
    momAng=np.array(momAng)
    momLin=np.array(momLin)
    #rMomentaneo=np.array(rMomentaneo)
    #aceleracoes=np.array(aceleracoes)

    #salvarEstadosNPZ(massas,estado[0],estado[2],estado[4],tempoSimulacao,momAng,momLin)


    #para facilitar no treinamento da rede, salvo todos os arrays em um arquivo compactado
    np.savez_compressed(f"simulacoesArtificiais/simulacoesTeste/simulacao3D_estruturaTeste{nome}.npz",
                        #massas=massas,trajetoria1=trajetoria1,trajetoria2=trajetoria2,trajetoria3=trajetoria3,tempo=tempo,dt=dt)
                        massas=massas,estado=estado,tempo=tempo,energia=energia,momAng=momAng,momLin=momLin,dt=dt)
    print(f"Arquivo salvo. Tamanho aproximado: {estado.nbytes/.33e3:.1f} KB")

'''def salvarEstadosNPZ(massas,trajetoria,tempo,energia,momAng,momLin,rMomentaneo,aceleracoes,nome,dt):
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
    np.savez_compressed(f"simulacoesArtificiais/simulacoesTeste/simulacao3D{nome}.npz",
                        massas=massas,trajetoria=trajetoria,tempo=tempo,energia=energia,momAng=momAng,momLin=momLin,rMomentaneo=rMomentaneo,aceleracoes=aceleracoes,dt=dt)
    print(f"Arquivo salvo. Tamanho aproximado: {trajetoria.nbytes/1e6:.1f} MB")'''

def yoshida4ordem(estado,dt,m1,m2,m3):#utiliza algumas vezes o velocity-verlet
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

    return np.concatenate([r1,v1,r2,v2,r3,v3])

def atualizaAceleracoes_posicoes(r1,r2,r3,m1,m2,m3):
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

    return a1,a2,a3

def calculaEnergiaDoSistema(estado,m1,m2,m3): #e momentos linear e angular
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
    #return m1*(r1[0]*v1[1] - r1[1]*v1[0]) + m2*(r2[0]*v2[1] - r2[1]*v2[0]) + m3*(r3[0]*v3[1] - r3[1]*v3[0])

def rodarSimulacao(seed):
    np.random.seed(seed)
    rMomentaneo=[]
    momLin=[]
    momAng=[]
    aceleracoes=[]
    trajetoria=[]
    energiaDoSistema=[]
    tAtual=0.0
    flag_colisao=0


    x1,y1,z1,vx1,vy1,vz1,\
    x2,y2,z2,vx2,vy2,vz2,\
    x3,y3,z3,vx3,vy3,vz3=estado0

    #alternado as massas (mas mantendo a massa total = 1)
    m1simulacao=np.random.rand()
    resto=1-m1simulacao
    m2simulacao=resto*np.random.rand()
    m3simulacao=1-m1simulacao-m2simulacao

    #alterando as posições e velocidades iniciais do sistema
    estadoAleatorio=np.array([rd(a) for a in[ #gera numeros aleatórios baseando-se nos valores origianais do estado0, sem alterá-los no molde
        x1,y1,z1,vx1,vy1,vz1,\
        x2,y2,z2,vx2,vy2,vz2,\
        x3,y3,z3,vx3,vy3,vz3
    ]])
    x1,y1,z1,vx1,vy1,vz1,x2,y2,z2,vx2,vy2,vz2,x3,y3,z3,vx3,vy3,vz3=estadoAleatorio

    #mantendo o momento linear do sistema, forçadamente, em ZERO        P_total=m1simulacao*v1+m2simulacao*v2+m3simulacao*v3       
    r1=np.array([x1,y1,z1])
    r2=np.array([x2,y2,z2])
    r3=np.array([x3,y3,z3])
    v1=np.array([vx1,vy1,vz1])
    v2=np.array([vx2,vy2,vz2])
    v3=np.array([vx3,vy3,vz3])
    
    V_cm=(m1simulacao*v1+m2simulacao*v2+m3simulacao*v3)/(m1simulacao+m2simulacao+m3simulacao)

    v1-=V_cm
    v2-=V_cm
    v3-=V_cm

    #colocando o centro de massa do sistema na origem da simulação, para facilitar o treinamento da rede (evita que ela tenha que aprender o DRIFT - o centro de massa estar em certa posição não afeta na evolução da simulação)
    R_cm=(m1simulacao*r1+m2simulacao*r2+m3simulacao*r3)/(m1simulacao+m2simulacao+m3simulacao)

    r1-=R_cm
    r2-=R_cm
    r3-=R_cm

    estado = np.array([
        r1[0],r1[1],r1[2],v1[0],v1[1],v1[2],
        r2[0],r2[1],r2[2],v2[0],v2[1],v2[2],
        r3[0],r3[1],r3[2],v3[0],v3[1],v3[2]
    ])

    saverCounter=0

    verificadorTamanho=0

    for i in range(steps):
        #tempoSimulacao.append(tAtual)
            
        #energiaDoSistema.append(calculaEnergiaDoSistema(estado,m1simulacao,m2simulacao,m3simulacao))

        #PARA EVITAR COLISÕES, QUE NÃO É O OBJETIVO DA REDE COMPREENDER, COLOCO ESSE GATILHO PARA EVITAR COLOCAR OS DADOS DA TRAJETÓRIA NO DATASET
        r12=np.sqrt((np.linalg.norm(np.array([estado[6],estado[7],estado[8]])-np.array([estado[0],estado[1],estado[2]])))**2 + epsilon**2)
        r13=np.sqrt((np.linalg.norm(np.array([estado[12],estado[13],estado[14]])-np.array([estado[0],estado[1],estado[2]])))**2 + epsilon**2)
        r23=np.sqrt((np.linalg.norm(np.array([estado[12],estado[13],estado[14]])-np.array([estado[6],estado[7],estado[8]])))**2 + epsilon**2)


        '''if min(r12,r13,r23)<1 or energiaDoSistema[-1]>0: #o [-1] pega o último elemento da lista, que seria o mais atual cálculo da energia do sistema
            print("\nCOLISAO DETECTADA ou SISTEMA HIPERBÓLICO, CANCELANDO SIMULAÇÃO")
            print("\nPASSO \n\n",i)
            flag_colisao=1
            break'''
        
        #trajetoria.append(np.append(estado.copy(),tAtual))#aqui o trajetoria é uma lista de arrays

        r1=estado[0:3]
        v1=estado[3:6]
        r2=estado[6:9]
        v2=estado[9:12]
        r3=estado[12:15]
        v3=estado[15:18]

        a1,a2,a3=atualizaAceleracoes_posicoes(r1,r2,r3,m1simulacao,m2simulacao,m3simulacao)
        aceleracoes.append(np.concatenate([a1,a2,a3]))

        if saverCounter%40==0:
            verificadorTamanho+=1
            tempoSimulacao.append(tAtual)
            energiaDoSistema.append(calculaEnergiaDoSistema(estado,m1simulacao,m2simulacao,m3simulacao))
            #trajetoria.append(np.append(estado.copy(),tAtual))#aqui o trajetoria é uma lista de arrays
            trajetoria.append(np.append(estado.copy(),tAtual/40))#aqui o trajetoria é uma lista de arrays
            momLin.append(calculaMomLin(m1simulacao,m2simulacao,m3simulacao,v1,v2,v3))
            momAng.append(calculaMomAng(m1simulacao,m2simulacao,m3simulacao,r1,r2,r3,v1,v2,v3))
            rMomentaneo.append(np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)+np.sqrt((np.linalg.norm(r3-r1))**2 + epsilon**2)+np.sqrt((np.linalg.norm(r2-r3))**2 + epsilon**2))

        saverCounter+=1

        #momLin.append(calculaMomLin(m1simulacao,m2simulacao,m3simulacao,v1,v2,v3))
        #momAng.append(calculaMomAng(m1simulacao,m2simulacao,m3simulacao,r1,r2,r3,v1,v2,v3))
        #rMomentaneo.append(np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)+np.sqrt((np.linalg.norm(r3-r1))**2 + epsilon**2)+np.sqrt((np.linalg.norm(r2-r3))**2 + epsilon**2))

        if min(r12,r13,r23)<1 or energiaDoSistema[-1]>0: #o [-1] pega o último elemento da lista, que seria o mais atual cálculo da energia do sistema
            print("\nCOLISAO DETECTADA ou SISTEMA HIPERBÓLICO, CANCELANDO SIMULAÇÃO")
            print("\nPASSO \n\n",i)
            flag_colisao=1
            break


        #progresso da simulação em porcentagem 
        if i % 10000 == 0 and i > 0:
            print(f"Passo {i}/{steps} ({100*i/steps:.1f}%)")
        
        #=============================================EVOLUÇÃO DO SISTEMA=============================================
        estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao,m3simulacao)
        tAtual+=dt
        
    #salva os dados da simulação se não houve colisão
    if(flag_colisao==0):  
        massas=[m1simulacao,m2simulacao,m3simulacao]
        
        trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos
        energiaDoSistema=np.array(energiaDoSistema)

        #salvarEstadosNPZ(massas,trajetoria,tempoSimulacao,energiaDoSistema,momAng,momLin,rMomentaneo,aceleracoes,seed,dt) #ESTOU SALVANDO TUDO ISSO, PORÉM EU PRECISO APENAS SALVAR AS POSIÇÕES NA TRAJETÓRIA (que contém as posições e velocidades), MASSAS E MOMENTOS
        
        salvarEstadosNPZ(massas,trajetoria,tempoSimulacao,energiaDoSistema,momAng,momLin,seed,dt)
        print("Quantidade de pontos salvos: ", verificadorTamanho)
        print("steps totais", steps)
        return seed

def converter_V_para_P(m1,m2,m3,estado): #usado após ser gerado pelo yoshida, no loop
    estadoAux=estado.copy()
    estadoAux[1] *= m1
    estadoAux[3] *= m2
    estadoAux[5] *= m3
    return estadoAux

def converter_P_para_V(m1,m2,m3,estado): #usado após ser salvo vetor de estados para retornar para velocidades para manter o cálculo da LOSS de maneira correta
    estadoAux=estado.copy()
    estadoAux[1] /= m1
    estadoAux[3] /= m2
    estadoAux[5] /= m3
    return estadoAux
#============================== VARIÁVEIS DA SIMULAÇÃO =================================

estado0=np.array([
    -10, -5, 6, .11, .5, .4,
     10, -10, 3, -.2, -.6, .8,
     5, 10, -8, -.55, 0 , .3
],dtype=float)

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
epsilon=1e-6  #0  #1e-5




#============================== SIMULAÇÃO =================================


if __name__ == "__main__":
    os.makedirs("simulacoesArtificiais/simulacoesTeste", exist_ok=True)
    NUM_SIMUL = 3

    with Pool(processes=os.cpu_count()) as pool:
        resultados = pool.map(rodarSimulacao, range(NUM_SIMUL))

    salvas    = [r for r in resultados if r is not None]
    descartadas = [r for r in resultados if r is None]
    print(f"\nConcluído: {len(salvas)} salvas, {len(descartadas)} descartadas.")