from multiprocessing import Pool
import os, numpy as np
from fisica2corpos import *
from fisica3corpos import *
from preparar_dataset import salvarEstadosNPZ,carregarSeedsUsadas,proximoBlocoSeeds,rd
from config import *


def rodarSimulacao3Corpos(seed):
    np.random.seed(seed)
    momLin=[]
    momAng=[]
    trajetoria=[]
    energiaDoSistema=[]
    tempoSimulacao=[]
    tAtual=0.0
    flag_colisao=0


    x1,y1,z1,vx1,vy1,vz1,\
    x2,y2,z2,vx2,vy2,vz2,\
    x3,y3,z3,vx3,vy3,vz3=ESTADO0_3_CORPOS

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

    verificadorTamanho=0
    saverCounter=0

    #motivoTermino=0 --> simulação completa     --> salvando ela inteira
    #motivoTermino=1 --> simulação com colisão  --> salvando até antes da margem de segurança
    #motivoTermino=2 --> simulação hiperbólica  --> salvando ela inteira
    motivoTermino=0


    for i in range(steps):
        #PARA EVITAR COLISÕES, QUE NÃO É O OBJETIVO DA REDE COMPREENDER, COLOCO ESSE GATILHO PARA EVITAR COLOCAR OS DADOS DA TRAJETÓRIA NO DATASET
        r12=np.sqrt((np.linalg.norm(np.array([estado[6],estado[7],estado[8]])-np.array([estado[0],estado[1],estado[2]])))**2 + epsilon**2)
        r13=np.sqrt((np.linalg.norm(np.array([estado[12],estado[13],estado[14]])-np.array([estado[0],estado[1],estado[2]])))**2 + epsilon**2)
        r23=np.sqrt((np.linalg.norm(np.array([estado[12],estado[13],estado[14]])-np.array([estado[6],estado[7],estado[8]])))**2 + epsilon**2)

        houveProximidade=min(r12,r13,r23)<1

        if houveProximidade:
            print(f"\nENCONTRO DETECTADO - seed {seed} - no passo {i}")
            motivoTermino=1
            break

        r1=estado[0:3]
        v1=estado[3:6]
        r2=estado[6:9]
        v2=estado[9:12]
        r3=estado[12:15]
        v3=estado[15:18]

        if saverCounter%SALVAR_A_CADA==0:
            verificadorTamanho+=1
            tempoSimulacao.append(tAtual)
            energiaDoSistema.append(calculaEnergiaDoSistema3Corpos(estado,m1simulacao,m2simulacao,m3simulacao))
            trajetoria.append(np.append(estado.copy(),tAtual))#aqui o trajetoria é uma lista de arrays
            momLin.append(calculaMomLin3Corpos(m1simulacao,m2simulacao,m3simulacao,v1,v2,v3))
            momAng.append(calculaMomAng3Corpos(m1simulacao,m2simulacao,m3simulacao,r1,r2,r3,v1,v2,v3))
            if energiaDoSistema[-1]>0:
                motivoTermino=2
        saverCounter+=1

        #progresso da simulação em porcentagem 
        if i % 10000 == 0 and i > 0:
            print(f"Passo {i}/{steps} ({100*i/steps:.1f}%)")
        
        #=============================================EVOLUÇÃO DO SISTEMA=============================================
        estado=yoshida4ordem3Corpos(estado,dt,m1simulacao,m2simulacao,m3simulacao)
        tAtual+=dt

    #salva os dados da simulação se não houve colisão
    if(flag_colisao==0):  
        massas=[m1simulacao,m2simulacao,m3simulacao]
        
        trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos
        energiaDoSistema=np.array(energiaDoSistema)

        if motivoTermino == 1 and len(trajetoria) > MARGEM_SEGURANCA:
            trajetoria=trajetoria[:-MARGEM_SEGURANCA]
            tempoSimulacao=tempoSimulacao[:-MARGEM_SEGURANCA]
            energiaDoSistema=energiaDoSistema[:-MARGEM_SEGURANCA]
            momAng=momAng[:-MARGEM_SEGURANCA]
            momLin=momLin[:-MARGEM_SEGURANCA]
        salvarEstadosNPZ("simulacoesArtificiais/simulacoes3C/",massas,trajetoria,tempoSimulacao,energiaDoSistema,momAng,momLin,seed,dt,motivoTermino)
        #print("Quantidade de pontos salvos: ", verificadorTamanho-margemDeSeguranca/40)
        #print("Steps totais", steps)
        
        return seed

def rodarSimulacao2Corpos(seed):
    np.random.seed(seed)
    momLin=[]
    momAng=[]
    trajetoria=[]
    energiaDoSistema=[]
    tempoSimulacao=[]
    tAtual=0.0
    flag_colisao=0


    x1,y1,z1,vx1,vy1,vz1,\
    x2,y2,z2,vx2,vy2,vz2=ESTADO0_2_CORPOS

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
    

    #motivoTermino=0 --> simulação completa     --> salvando ela inteira
    #motivoTermino=1 --> simulação com colisão  --> salvando até antes da margem de segurança
    #motivoTermino=2 --> simulação hiperbólica  --> salvando ela inteira
    motivoTermino=0

    for i in range(steps):
        #PARA EVITAR COLISÕES, QUE NÃO É O OBJETIVO DA REDE COMPREENDER, COLOCO ESSE GATILHO PARA EVITAR COLOCAR OS DADOS DA TRAJETÓRIA NO DATASET --> agora deixamos todas as simulações
        r12=np.sqrt((np.linalg.norm(np.array([estado[6],estado[7],estado[8]])-np.array([estado[0],estado[1],estado[2]])))**2 + epsilon**2)

        houveProximidade=r12<1

        if houveProximidade:
            print(f"\nENCONTRO DETECTADO - seed {seed} - no passo {i}")
            motivoTermino=1
            break

        r1=estado[0:3]
        v1=estado[3:6]
        r2=estado[6:9]
        v2=estado[9:12]

        if saverCounter%SALVAR_A_CADA==0:
            verificadorTamanho+=1
            tempoSimulacao.append(tAtual)
            energiaDoSistema.append(calculaEnergiaDoSistema2Corpos(estado,m1simulacao,m2simulacao))
            trajetoria.append(np.append(estado.copy(),tAtual))#aqui o trajetoria é uma lista de arrays
            momLin.append(calculaMomLin2Corpos(m1simulacao,m2simulacao,v1,v2))
            momAng.append(calculaMomAng2Corpos(m1simulacao,m2simulacao,r1,r2,v1,v2))
            if energiaDoSistema[-1]>0:
                motivoTermino=2

        saverCounter+=1

        #progresso da simulação em porcentagem 
        if i % 10000 == 0 and i > 0:
            print(f"Passo {i}/{steps} ({100*i/steps:.1f}%)")

        #=============================================EVOLUÇÃO DO SISTEMA=============================================
        estado=yoshida4ordem2Corpos(estado,dt,m1simulacao,m2simulacao)
        tAtual+=dt
        
    #salva os dados da simulação se não houve colisão --> agora salva
    if(flag_colisao==0):  
        massas=[m1simulacao,m2simulacao]
        
        trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos
        energiaDoSistema=np.array(energiaDoSistema)
        
        if motivoTermino == 1 and len(trajetoria) > MARGEM_SEGURANCA:
            trajetoria=trajetoria[:-MARGEM_SEGURANCA]
            tempoSimulacao=tempoSimulacao[:-MARGEM_SEGURANCA]
            energiaDoSistema=energiaDoSistema[:-MARGEM_SEGURANCA]
            momAng=momAng[:-MARGEM_SEGURANCA]
            momLin=momLin[:-MARGEM_SEGURANCA]

        salvarEstadosNPZ("simulacoesArtificiais/simulacoes2C/",massas,trajetoria,tempoSimulacao,energiaDoSistema,momAng,momLin,seed,dt,motivoTermino)

        return seed

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




#============================== SIMULAÇÃO =================================
#aqui a seleção indica qual dataset será gerado
#selecao = 0  -->   3 CORPOS
#selecao = 1  -->   2 CORPOS
selecao=1

if selecao==0: 
    total=TOTALSIMULACOES_3_CORPOS
    funcaoSimulacao=rodarSimulacao3Corpos
    pastaSaida=PASTA_SAIDA_3_CORPOS
    pastaSeeds=PASTA_SEEDS_3_CORPOS
else: 
    total=TOTALSIMULACOES_2_CORPOS
    funcaoSimulacao=rodarSimulacao2Corpos
    pastaSaida=PASTA_SAIDA_2_CORPOS
    pastaSeeds=PASTA_SEEDS_2_CORPOS


numeroDeExistentes=len(carregarSeedsUsadas(pastaSeeds))
if __name__ == "__main__":
    while(total>numeroDeExistentes):
        os.makedirs(pastaSaida, exist_ok=True)

        NUM_SIMUL = (total-numeroDeExistentes)
        seedsDesteLote=proximoBlocoSeeds(NUM_SIMUL,pastaSeeds)
        print("Gerando as seeds ", {seedsDesteLote[0]}, " até ", {seedsDesteLote[-1]})

        with Pool(processes=os.cpu_count()) as pool:
            resultados = pool.map(funcaoSimulacao, seedsDesteLote)

        salvas    = [r for r in resultados if r is not None]
        with open(pastaSeeds,"a") as file:
            for seed in salvas: 
                file.write(f"{seed}\n")
        numeroDeExistentes+=len(salvas)
        #print(f"\nConcluído: {len(salvas)} salvas neste lote de ",NUM_SIMUL," simuações.\nTOTAL DE SIMULAÇÕES GERADAS: ",numeroDeExistentes)
        #if len(salvas) <= NUM_SIMUL/7: NUM_SIMUL = (total-numeroDeExistentes)*7 #aqui é vezes sete a quantidade de simulações que ainda preciso pois é a proporção que encontrei de simulações geradas X simulações não colisionais ou hiperbólicas


