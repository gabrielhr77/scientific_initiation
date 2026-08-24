import numpy as np, matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import time


def salvarEstadosNPZ(massas,estado,tempo,energia,momAng,momLin,nome,dt,motivoTermino):
    #convertendo para array caso ainda não seja
    massas=np.array(massas)
    estado=estado
    tempo=np.array(tempo)
    energia=np.array(energia)
    momAng=np.array(momAng)
    momLin=np.array(momLin)
    motivoTermino=motivoTermino

    np.savez_compressed(f"simulacoesArtificiais2C/simulacoesTeste/simulacao3D_estruturaTeste2C{nome}.npz",
                        massas=massas,estado=estado,tempo=tempo,energia=energia,momAng=momAng,momLin=momLin,dt=dt,motivoTermino=motivoTermino)
    print(f"Arquivo da seed {nome} salvo. Tamanho aproximado: {estado.nbytes/1024:.1f} KB")

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

def rodarSimulacao(seed):
    np.random.seed(seed)
    momLin=[]
    momAng=[]
    trajetoria=[]
    energiaDoSistema=[]
    tempoSimulacao=[]
    tAtual=0.0
    flag_colisao=0


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
    x1,y1,z1,vx1,vy1,vz1,x2,y2,z2,vx2,vy2,vz2=estadoAleatorio

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

    saverCounter=0
    verificadorTamanho=0
    margemDeSeguranca=100  #em steps fica 40*100 (uma vez que estou salvando uma vez a cada 40 passos dados)

    #motivoTermino=0 --> simulação completa     --> salvando ela inteira
    #motivoTermino=1 --> simulação com colisão  --> salvando até antes da margem de segurança
    #motivoTermino=2 --> simulação hiperbólica  --> salvando ela inteira
    motivoTermino=0

    for i in range(steps):
        #PARA EVITAR COLISÕES, QUE NÃO É O OBJETIVO DA REDE COMPREENDER, COLOCO ESSE GATILHO PARA EVITAR COLOCAR OS DADOS DA TRAJETÓRIA NO DATASET --> agora deixamos todas as simulações
        r12=np.sqrt((np.linalg.norm(np.array([estado[6],estado[7],estado[8]])-np.array([estado[0],estado[1],estado[2]])))**2 + epsilon**2)

        houveProximidade=r12<1
        houveHiperbolismo=energiaDoSistema[-1]>0 if energiaDoSistema else False #testa primeiro "energiaDoSistema", que se nao for vazio retorna true, depois testa "energiaDoSistema[-1]>0", se der verdadeiro, fica como TRUE, senão fica FALSE

        if houveProximidade:
            print(f"\nENCONTRO DETECTADO - seed {seed} - no passo {i}")
            motivoTermino=1
            break
        elif houveHiperbolismo:
            print(f"\nSISTEMA HIPERBÓLICO DETECTADO - seed {seed} - passo {i}")
            motivoTermino=2

        r1=estado[0:3]
        v1=estado[3:6]
        r2=estado[6:9]
        v2=estado[9:12]

        if saverCounter%40==0:
            verificadorTamanho+=1
            tempoSimulacao.append(tAtual)
            energiaDoSistema.append(calculaEnergiaDoSistema(estado,m1simulacao,m2simulacao))
            trajetoria.append(np.append(estado.copy(),tAtual))#aqui o trajetoria é uma lista de arrays
            momLin.append(calculaMomLin(m1simulacao,m2simulacao,v1,v2))
            momAng.append(calculaMomAng(m1simulacao,m2simulacao,r1,r2,v1,v2))
            if energiaDoSistema[-1]>0:
                motivoTermino=2

        saverCounter+=1

        #progresso da simulação em porcentagem 
        if i % 10000 == 0 and i > 0:
            print(f"Passo {i}/{steps} ({100*i/steps:.1f}%)")

        #=============================================EVOLUÇÃO DO SISTEMA=============================================
        estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao)
        tAtual+=dt
        
    #salva os dados da simulação se não houve colisão --> agora salva
    if(flag_colisao==0):  
        massas=[m1simulacao,m2simulacao]
        
        trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos
        energiaDoSistema=np.array(energiaDoSistema)
        
        if motivoTermino == 1 and len(trajetoria) > margemDeSeguranca:
            trajetoria=trajetoria[:-margemDeSeguranca]
            tempoSimulacao=tempoSimulacao[:-margemDeSeguranca]
            energiaDoSistema=energiaDoSistema[:-margemDeSeguranca]
            momAng=momAng[:-margemDeSeguranca]
            momLin=momLin[:-margemDeSeguranca]

        salvarEstadosNPZ(massas,trajetoria,tempoSimulacao,energiaDoSistema,momAng,momLin,seed,dt,motivoTermino)
        print("Quantidade de pontos salvos: ", verificadorTamanho-margemDeSeguranca/40)
        print("steps totais", steps)
        return seed

def converter_V_para_P(m1,m2,estado): #usado após ser gerado pelo yoshida, no loop
    estadoAux=estado.copy()
    estadoAux[3:6] *= m1
    estadoAux[9:12] *= m2
    return estadoAux

def converter_P_para_V(m1,m2,estado): #usado após ser salvo vetor de estados para retornar para velocidades para manter o cálculo da LOSS de maneira correta
    estadoAux=estado.copy()
    estadoAux[3:6] /= m1
    estadoAux[9:12] /= m2
    return estadoAux

def carregarSeedsUsadas(caminho="seedsUsadas2C.txt"):
    if not os.path.exists(caminho):
        return set() #o que seria esse SET()?
    with open(caminho) as file:
        return {int(linha.strip()) for linha in file if linha.strip()} #retorna um vetor com as linhas salvas

def proximoBlocoSeeds(qtdd,caminho="seedsUsadas2C.txt"):
    usadas=carregarSeedsUsadas(caminho)
    proximo=(max(usadas)+1) if usadas else 0
    return list(range(proximo, proximo+qtdd))


#============================== VARIÁVEIS DA SIMULAÇÃO =================================
estado0=np.array([
    -10, -5, 6, .11, .5, .4,
     10, -10, 3, -.2, -.6, .8,
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
total=20
numeroDeExistentes=len(carregarSeedsUsadas("seedsUsadas2C.txt"))
if __name__ == "__main__":
    NUM_SIMUL = (total-numeroDeExistentes)#*7
    while(total>numeroDeExistentes):
        os.makedirs("simulacoesArtificiais2C/simulacoesTeste", exist_ok=True)

        seedsDesteLote=proximoBlocoSeeds(NUM_SIMUL)
        print("Gerando as seeds ", {seedsDesteLote[0]}, " até ", {seedsDesteLote[-1]})

        with Pool(processes=os.cpu_count()) as pool:
            resultados = pool.map(rodarSimulacao, seedsDesteLote)

        salvas    = [r for r in resultados if r is not None]
        with open("seedsUsadas2C.txt","a") as file:
            for seed in salvas: 
                file.write(f"{seed}\n")
        numeroDeExistentes+=len(salvas)
        print(f"\nConcluído: {len(salvas)} salvas neste lote de ",NUM_SIMUL," simuações.\nTOTAL DE SIMULAÇÕES GERADAS: ",numeroDeExistentes)
        #if len(salvas) <= NUM_SIMUL/7: NUM_SIMUL = (total-numeroDeExistentes)*7 #aqui é vezes sete a quantidade de simulações que ainda preciso pois é a proporção que encontrei de simulações geradas X simulações não colisionais ou hiperbólicas