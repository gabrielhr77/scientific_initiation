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
    
    #garantindo que a razão de massas seja menor que seis, então evita a presença de sistemas hierárquicos, evitando um sistema
    # de 3 corpos que aja como um sistema binário com 1 corpo com 90% da massa, outro com 9% e outro com menos de 1% --> é para 
    #ser um experimento de extrapolação das capacidades da rede, então treino ela para que generalize em casos mais 
    #complexos (pois com massas parecidas o sistema é MUITO mais caótico) para depois ver sua capacidade de adaptação ao inédito,
    # podendo isso ser um sistema hierárquico com razão grande entre os corpos
    if max(m1simulacao, m2simulacao, m3simulacao) > RAZAO_MASSA_MAX * min(m1simulacao, m2simulacao, m3simulacao):
        return None

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

    #aqui estava o problema que não estava resolvido para a geração, a escala variava, entre as simulações, mais ou menos umas 1000x, com 1/4 das simulações estando ligaddas gravitacionalmente         
    #aqui vou fazer o seguinte:
    #   medir a energia do sorteio do estado --> ajustar as velocidades para uma RAZÃO VIRIAL determinada (entre 0.25 e 0.9) --> escalar tudo para E=-1/4
    #   1- definir arrays de massas, posições e velocidades 
    massasFiltro=np.array([m1simulacao,m2simulacao,m3simulacao])
    posicoesFiltro=np.array([r1,r2,r3])
    velocidadesFiltro=np.array([v1,v2,v3])
    #   2- definir distâncias entre os pares de corpos
    d12,d13,d23=np.linalg.norm(posicoesFiltro[0]-posicoesFiltro[1]),np.linalg.norm(posicoesFiltro[0]-posicoesFiltro[2]),np.linalg.norm(posicoesFiltro[1]-posicoesFiltro[2])
    #   3- definir a energia potencial e a cinética
    K0=0.5*np.sum(massasFiltro*(velocidadesFiltro**2).sum(1))
    U0=-G*(m1simulacao*m2simulacao/d12+m1simulacao*m3simulacao/d13+m2simulacao*m3simulacao/d23)
    #   4- sortear a razão virial entre 0.25 e 0.9 --> o que seria a razão de virial? se o sistema de 3 corpos está em equilíbrio estático estável
    #       a razão instantânea entre 2*<T> (energia cinética média) e <U> (energia potencial média) deve ficar próxima de 1 ( q(t) = 2T(t)/|U(t)| ≈ 1),
    #       e caso se afaste disso significa que o sistema está se DESINTEGRANDO com um corpo ganhando velocidade de escape do sistema com as transferências 
    #       de energia durante os encontros próximos --> para garantir que o corpo foi ejetado eu tenho que ver essa razão e calcular a energia do corpo para 
    #       ver se ela fica POSITIVA com a razão virial se estabilizando em torno de 1.
    #       vale lembrar que um sistema ligado tem q<1 e quando está em q≈0.5
    q=np.random.uniform(qMin3C,qMax3C)
    #   5- ajustar as velocidades de acordo com a razão virial
    fatorVirial=np.sqrt(q*abs(U0)/K0)
    v1,v2,v3=v1*fatorVirial,v2*fatorVirial,v3*fatorVirial
    #   6- achar a energia resultante depois do ajuste com a razão virial
    E0=q*abs(U0)+U0
    #   7- escalar para E= -1/4 --> levo todo sorteio à mesma energia E = -1/4 sem mudar a forma da órbita uma vez que o sistema inteiro pode ser esticado ou
    #       encolhido sem perder a dinâmica original --> como G=1 e M=1 tenfo que a energia potencial é proporcional a 1/r (se todas as distâncias são multiplicadas
    #       por LAMBDA, U fica dividido por LAMBDA); a energia cinética é proporcional a v² (se todas as velocidades são divididas por raiz(LAMBDA), K fica dividido 
    #       por LAMBDA também) -> como K e U são divididas por LAMBDA, a energia total E também vaidicar, então escolho LAMBDA=E0/E_alvo ==> E0/LAMBDA=E_alvo=-1/4: 
    #       caso E0= -0.01, LAMBDA=0.01/0.25 = 0.04 -> as posições são multiplicadas por 0.04 (sistema encolhe 25x), as vel são divididas por raiz(0.04) (vel multiplicadas por 5),
    #       U sobe 25x e 5 sobe 25x tmb, logo a energia vai de -0.01 para -0.025 --> isso fixa a escala com o dt=0.05 entre amostras passando a significar a mesma 
    #       fração da órbita em todos os sorteios
    lambda0=E0/E_ALVO_3C
    r1,r2,r3=r1*lambda0,r2*lambda0,r3*lambda0
    v1,v2,v3=v1/np.sqrt(lambda0),v2/np.sqrt(lambda0),v3/np.sqrt(lambda0)
    #   8- rejeitar começos ruins onde as distâncias entre pares seja menor que um limiar definido como inaceitável
    if min(np.linalg.norm(r1-r2),np.linalg.norm(r1-r3),np.linalg.norm(r2-r3))<1.5*R_ENCONTRO_3C:   
        return None
    
    
    
    
    #---------------------------------------------------

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

    

    for i in range(STEPS_3C):
        #PARA EVITAR COLISÕES, QUE NÃO É O OBJETIVO DA REDE COMPREENDER, COLOCO ESSE GATILHO PARA EVITAR COLOCAR OS DADOS DA TRAJETÓRIA NO DATASET
        r12=np.sqrt((np.linalg.norm(np.array([estado[6],estado[7],estado[8]])-np.array([estado[0],estado[1],estado[2]])))**2 + epsilon**2)
        r13=np.sqrt((np.linalg.norm(np.array([estado[12],estado[13],estado[14]])-np.array([estado[0],estado[1],estado[2]])))**2 + epsilon**2)
        r23=np.sqrt((np.linalg.norm(np.array([estado[12],estado[13],estado[14]])-np.array([estado[6],estado[7],estado[8]])))**2 + epsilon**2)

        houveProximidade=min(r12,r13,r23)<R_ENCONTRO_3C

        if houveProximidade:
            #print(f"\nENCONTRO DETECTADO - seed {seed} - no passo {i}")
            motivoTermino=1
            break

        r1=estado[0:3]
        v1=estado[3:6]
        r2=estado[6:9]
        v2=estado[9:12]
        r3=estado[12:15]
        v3=estado[15:18]

        if saverCounter%SALVAR_A_CADA_3C==0:
            verificadorTamanho+=1
            tempoSimulacao.append(tAtual)
            energiaDoSistema.append(calculaEnergiaDoSistema3Corpos(estado,m1simulacao,m2simulacao,m3simulacao))
            trajetoria.append(np.append(estado.copy(),tAtual))#gaqui o trajetoria é uma lista de arrays
            momLin.append(calculaMomLin3Corpos(m1simulacao,m2simulacao,m3simulacao,v1,v2,v3))
            momAng.append(calculaMomAng3Corpos(m1simulacao,m2simulacao,m3simulacao,r1,r2,r3,v1,v2,v3))
            if max(r12,r13,r23)>R_FUGA_3C:
                motivoTermino=2
                break
        saverCounter+=1

        #progresso da simulação em porcentagem 
        '''if i % 10000 == 0 and i > 0:
            print(f"Passo {i}/{STEPS_3C} ({100*i/STEPS_3C:.1f}%)")
        '''
        #=============================================EVOLUÇÃO DO SISTEMA=============================================
        estado=yoshida4ordem3Corpos(estado,DT_3C,m1simulacao,m2simulacao,m3simulacao)
        tAtual+=DT_3C

    #salva os dados da simulação se não houve colisão
    if(flag_colisao==0 and motivoTermino!=2):       #aqui só vai salvar as simulações que não forem hiperbólicas e as que houve colisão até MARGEM_SEGURANCA antes da colisão
        massas=[m1simulacao,m2simulacao,m3simulacao]
        
        trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos
        energiaDoSistema=np.array(energiaDoSistema)
        tempoSimulacao=list(tempoSimulacao)

        if motivoTermino == 1 and len(trajetoria) > MARGEM_3C:
            trajetoria=trajetoria[:-MARGEM_3C]
            tempoSimulacao=tempoSimulacao[:-MARGEM_3C]
            energiaDoSistema=energiaDoSistema[:-MARGEM_3C]
            momAng=momAng[:-MARGEM_3C]
            momLin=momLin[:-MARGEM_3C]
        if len(trajetoria)<N_MIN_3C:
            return None
        salvarEstadosNPZ("simulacoesArtificiais/simulacoes3C/",massas,trajetoria,tempoSimulacao,energiaDoSistema,momAng,momLin,seed,DT_3C*SALVAR_A_CADA_3C,motivoTermino)
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

    m1simulacao=np.random.rand()
    m2simulacao=1-m1simulacao

    #alterando as posições e velocidades iniciais do sistema
    estadoAleatorio=np.array([rd(a) for a in[ #gera numeros aleatórios baseando-se nos valores origianais do estado0, sem alterá-los no molde
        x1,y1,z1,vx1,vy1,vz1,\
        x2,y2,z2,vx2,vy2,vz2
    ]])
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

    #aqui estava o problema que não estava resolvido para a geração, a escala variava, entre as simulações, mais ou menos umas 1000x, com 1/4 das simulações estando ligaddas gravitacionalmente         
    #aqui vou fazer o seguinte:
    #   medir a energia do sorteio do estado --> ajustar as velocidades para uma RAZÃO VIRIAL determinada (entre 0.25 e 0.9) --> escalar tudo para E=-1/4
    #   1- definir arrays de massas, posições e velocidades 
    massasFiltro=np.array([m1simulacao,m2simulacao])
    posicoesFiltro=np.array([r1,r2])
    velocidadesFiltro=np.array([v1,v2])
    #   2- definir distâncias entre os pares de corpos
    d12=np.linalg.norm(posicoesFiltro[0]-posicoesFiltro[1])
    #   3- definir a energia potencial e a cinética
    K0=0.5*np.sum(massasFiltro*(velocidadesFiltro**2).sum(1))
    U0=-G*(m1simulacao*m2simulacao/d12)
    #   4- sortear a razão virial entre 0.25 e 0.9 --> o que seria a razão de virial? se o sistema de 3 corpos está em equilíbrio estático estável
    #       a razão instantânea entre 2*<T> (energia cinética média) e <U> (energia potencial média) deve ficar próxima de 1 ( q(t) = 2T(t)/|U(t)| ≈ 1),
    #       e caso se afaste disso significa que o sistema está se DESINTEGRANDO com um corpo ganhando velocidade de escape do sistema com as transferências 
    #       de energia durante os encontros próximos --> para garantir que o corpo foi ejetado eu tenho que ver essa razão e calcular a energia do corpo para 
    #       ver se ela fica POSITIVA com a razão virial se estabilizando em torno de 1.
    #       vale lembrar que um sistema ligado tem q<1 e quando está em q≈0.5
    q=np.random.uniform(qMin2C,qMax2C)
    #   5- ajustar as velocidades de acordo com a razão virial
    fatorVirial=np.sqrt(q*abs(U0)/K0)
    v1,v2=v1*fatorVirial,v2*fatorVirial
    #   6- achar a energia resultante depois do ajuste com a razão virial
    E0=q*abs(U0)+U0
    #   7- escalar para E= -1/4 --> levo todo sorteio à mesma energia E = -1/4 sem mudar a forma da órbita uma vez que o sistema inteiro pode ser esticado ou
    #       encolhido sem perder a dinâmica original --> como G=1 e M=1 tenfo que a energia potencial é proporcional a 1/r (se todas as distâncias são multiplicadas
    #       por LAMBDA, U fica dividido por LAMBDA); a energia cinética é proporcional a v² (se todas as velocidades são divididas por raiz(LAMBDA), K fica dividido 
    #       por LAMBDA também) -> como K e U são divididas por LAMBDA, a energia total E também vaidicar, então escolho LAMBDA=E0/E_alvo ==> E0/LAMBDA=E_alvo=-1/4: 
    #       caso E0= -0.01, LAMBDA=0.01/0.25 = 0.04 -> as posições são multiplicadas por 0.04 (sistema encolhe 25x), as vel são divididas por raiz(0.04) (vel multiplicadas por 5),
    #       U sobe 25x e 5 sobe 25x tmb, logo a energia vai de -0.01 para -0.025 --> isso fixa a escala com o dt=0.05 entre amostras passando a significar a mesma 
    #       fração da órbita em todos os sorteios
    lambda0=E0/E_ALVO_2C
    r1,r2=r1*lambda0,r2*lambda0
    v1,v2=v1/np.sqrt(lambda0),v2/np.sqrt(lambda0)
    #   8- rejeitar começos ruins onde as distâncias entre pares seja menor que um limiar definido como inaceitável
    if np.linalg.norm(r1-r2)<1.5*R_ENCONTRO_2C:   
        return None
    
    
    
    
    #---------------------------------------------------

    estado = np.array([
        r1[0],r1[1],r1[2],v1[0],v1[1],v1[2],
        r2[0],r2[1],r2[2],v2[0],v2[1],v2[2]
    ])

    verificadorTamanho=0
    saverCounter=0

    #motivoTermino=0 --> simulação completa     --> salvando ela inteira
    #motivoTermino=1 --> simulação com colisão  --> salvando até antes da margem de segurança
    #motivoTermino=2 --> simulação hiperbólica  --> salvando ela inteira
    motivoTermino=0

    

    for i in range(STEPS_2C):
        #PARA EVITAR COLISÕES, QUE NÃO É O OBJETIVO DA REDE COMPREENDER, COLOCO ESSE GATILHO PARA EVITAR COLOCAR OS DADOS DA TRAJETÓRIA NO DATASET
        r12=np.sqrt((np.linalg.norm(np.array([estado[6],estado[7],estado[8]])-np.array([estado[0],estado[1],estado[2]])))**2 + epsilon**2)
        
        houveProximidade=r12<R_ENCONTRO_2C

        if houveProximidade:
            #print(f"\nENCONTRO DETECTADO - seed {seed} - no passo {i}")
            motivoTermino=1
            break

        r1=estado[0:3]
        v1=estado[3:6]
        r2=estado[6:9]
        v2=estado[9:12]

        if saverCounter%SALVAR_A_CADA_2C==0:
            verificadorTamanho+=1
            tempoSimulacao.append(tAtual)
            energiaDoSistema.append(calculaEnergiaDoSistema2Corpos(estado,m1simulacao,m2simulacao))
            trajetoria.append(np.append(estado.copy(),tAtual))#gaqui o trajetoria é uma lista de arrays
            momLin.append(calculaMomLin2Corpos(m1simulacao,m2simulacao,v1,v2))
            momAng.append(calculaMomAng2Corpos(m1simulacao,m2simulacao,r1,r2,v1,v2))
            if r12>R_FUGA_2C:
                motivoTermino=2
                break
        saverCounter+=1

        #progresso da simulação em porcentagem 
        '''if i % 10000 == 0 and i > 0:
            print(f"Passo {i}/{STEPS_3C} ({100*i/STEPS_3C:.1f}%)")
        '''
        #=============================================EVOLUÇÃO DO SISTEMA=============================================
        estado=yoshida4ordem2Corpos(estado,DT_2C,m1simulacao,m2simulacao)
        tAtual+=DT_2C

    #salva os dados da simulação se não houve colisão
    if(flag_colisao==0 and motivoTermino!=2):       #aqui só vai salvar as simulações que não forem hiperbólicas e as que houve colisão até MARGEM_SEGURANCA antes da colisão
        massas=[m1simulacao,m2simulacao]
        
        trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos
        energiaDoSistema=np.array(energiaDoSistema)
        tempoSimulacao=list(tempoSimulacao)

        if motivoTermino == 1 and len(trajetoria) > MARGEM_2C:
            trajetoria=trajetoria[:-MARGEM_2C]
            tempoSimulacao=tempoSimulacao[:-MARGEM_2C]
            energiaDoSistema=energiaDoSistema[:-MARGEM_2C]
            momAng=momAng[:-MARGEM_2C]
            momLin=momLin[:-MARGEM_2C]
        if len(trajetoria)<N_MIN_2C:
            return None
        salvarEstadosNPZ("simulacoesArtificiais/simulacoes2C/",massas,trajetoria,tempoSimulacao,energiaDoSistema,momAng,momLin,seed,DT_2C*SALVAR_A_CADA_2C,motivoTermino)
        #print("Quantidade de pontos salvos: ", verificadorTamanho-margemDeSeguranca/40)
        #print("Steps totais", steps)
        
        return seed
'''
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
'''

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
selecao=0

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
        NUM_SIMUL = 10*(total-numeroDeExistentes)#400*(total-numeroDeExistentes)
        seedsDesteLote=proximoBlocoSeeds(NUM_SIMUL,pastaSeeds)
        #print(seedsDesteLote)
        print("Gerando as seeds ", {seedsDesteLote[0]}, " até ", {seedsDesteLote[-1]})

        with Pool(processes=os.cpu_count()) as pool:
            resultados = pool.map(funcaoSimulacao, seedsDesteLote)

        while resultados==[None]*NUM_SIMUL:
            print("NUM_SIM antes: ",NUM_SIMUL)
            NUM_SIMUL*=2
            seedsDesteLote=proximoBlocoSeeds(NUM_SIMUL,pastaSeeds)
            print("Gerando as seeds ", {seedsDesteLote[0]}, " até ", {seedsDesteLote[-1]})
            print("NUM_SIM depois: ",NUM_SIMUL)
            with Pool(processes=os.cpu_count()) as pool:
                resultados = pool.map(funcaoSimulacao, seedsDesteLote)

        salvas    = [r for r in resultados if r is not None]
        print(salvas)
        with open(pastaSeeds,"a") as file:
            for seed in salvas: 
                file.write(f"{seed}\n")
        numeroDeExistentes+=len(salvas)
'''
        os.makedirs(pastaSaida, exist_ok=True)

        NUM_SIMUL = (total-numeroDeExistentes)
        #aqui pego o NUM_SIML e peço no mínimo 5 vezes o q precisa, uma vez que é MUITO pequena a proporção de trajetórias que serõ aceitas
        numeroSimulacoesCompensada=NUM_SIMUL*5
        seedsDesteLote=proximoBlocoSeeds(numeroSimulacoesCompensada,pastaSeeds)
        #print(seedsDesteLote)
        print("Gerando as seeds ", {seedsDesteLote[0]}, " até ", {seedsDesteLote[-1]})

        with Pool(processes=os.cpu_count()) as pool:
            resultados = pool.map(funcaoSimulacao, seedsDesteLote)

        while resultados==[None]*NUM_SIMUL:
            print("NUM_SIM antes: ",NUM_SIMUL)
            NUM_SIMUL*=2
            numeroSimulacoesCompensada=NUM_SIMUL*5
            seedsDesteLote=proximoBlocoSeeds(numeroSimulacoesCompensada,pastaSeeds)
            print("Gerando as seeds ", {seedsDesteLote[0]}, " até ", {seedsDesteLote[-1]})
            print("NUM_SIM depois: ",numeroSimulacoesCompensada)
            with Pool(processes=os.cpu_count()) as pool:
                resultados = pool.map(funcaoSimulacao, seedsDesteLote)

        salvas    = [r for r in resultados if r is not None]
        print(salvas)
        with open(pastaSeeds,"a") as file:
            for seed in salvas: 
                file.write(f"{seed}\n")
        numeroDeExistentes+=len(salvas)
        #print("NUMERO DE SEEDS EXISTENTES",numeroDeExistentes,"\nTOTAL PEDIDO: ",total,"\nQUANTAS FALTAM: ",NUM_SIMUL)
        #print("SEED INICIAL DA PROXIMA JANELA: ",seedsDesteLote[0])
        #print(f"\nConcluído: {len(salvas)} salvas neste lote de ",NUM_SIMUL," simuações.\nTOTAL DE SIMULAÇÕES GERADAS: ",numeroDeExistentes)
        #if len(salvas) <= NUM_SIMUL/7: NUM_SIMUL = (total-numeroDeExistentes)*7 #aqui é vezes sete a quantidade de simulações que ainda preciso pois é a proporção que encontrei de simulações geradas X simulações não colisionais ou hiperbólicas
        '''