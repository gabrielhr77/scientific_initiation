import numpy as np, matplotlib.pyplot as plt
import pandas as pd


def estadoAtual(estado,m1,m2):
    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2,t=estado #pareando as variáveis com seus respectivos valores, onde x1=estado[0]
    #vetores posição
    r1=np.array([x1,y1])
    r2=np.array([x2,y2])

    #distância
    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)#traz a norma da matriz/vetor (distância entre os dois pontos/vetores posição) e o sqrt com a soma do 1e-6 garante que a distância não será 0 para evitar erro em divisão

    #acelerações
    a1=G*m2*(r2-r1)/r12**3
    a2=G*m1*(r1-r2)/r12**3

    #retornando as derivadas masi o dt/dt=1
    return np.array([vx1,vy1,a1[0],a1[1],
                     vx2,vy2,a2[0],a2[1],
                     1])

def extrair_horizons(arquivoEntrada):
    dados={
        'X':[],
        'Y':[],
        'VX':[],
        'VY':[]
    }    #dicionario
    permitidos=['X','Y','VX','VY']
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
    VX=np.array(dados["VX"])
    VY=np.array(dados["VY"])
   
    np.savez_compressed(f"simulacoesArtificiais/horizons{nome}.npz",X=X,Y=Y,VX=VX,VY=VY)
    print(f"Arquivo horizons{nome} salvo. Tamanho aproximado: {(X.nbytes+Y.nbytes+VX.nbytes+VY.nbytes)/1024:.2f} KB")

#função para carregar os dados depois
def carregarDadosHorizonsNPZ(nome):
    dados=np.load(f"simulacoesArtificiais/horizons{nome}.npz")
    return (dados["X"],dados["Y"],dados["VX"],dados["VY"])

def salvarEstadosNPZ(massas,trajetoria,tempo,energia,momAng,momLin,r_min,nome):
    #convertendo para array caso ainda não seja
    massas=np.array(massas)
    trajetoria=np.array(trajetoria)
    tempo=np.array(tempo)
    energia=np.array(energia)
    momAng=np.array(momAng)
    momLin=np.array(momLin)
    r_min=np.array(r_min)

    #para facilitar no treinamento da rede, salvo todos os arrays em um arquivo compactado
    np.savez_compressed(f"simulacoesArtificiais/simulacao2C{nome}.npz",
                        massas=massas,trajetoria=trajetoria,tempo=tempo,energia=energia,momAng=momAng,momLin=momLin,r_min=r_min)
    print(f"Arquivo salvo. Tamanho aproximado: {trajetoria.nbytes/1e6:.1f} MB")


#função para carregar os dados depois
def carregarEstadosNPZ(nome):
    dados=np.load(f"simulacoesArtificiais/simulacao2C{nome}.npz")
    return dados['trajetoria']

def yoshida4ordem(estado,dt,m1,m2):#utiliza algumas vezes o velocity-verlet
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


#VERSÃO QUE NÃO CHAMA O NP.LINALOG.NORM, SÓ CONTENDO POUCOS SQRT E MULTIPLICAÇÕES, SEM DIVISÕES (que são mais lentas que as multiplicações)
def atualizaAceleracoes_posicoes(r1,r2,m1,m2):
    #vetor posição
    pr12=r2-r1
  
    #radicando
    rr12=pr12[0]**2 + pr12[1]**2 + epsilon**2

    #fazendo o inverso para poupar algumas divisões

    inv_r12=1/(rr12*np.sqrt(rr12))

    a1=G*(m2*pr12*inv_r12)
    a2=G*(-m1*pr12*inv_r12)

    return a1,a2


def calculaEnergiaDoSistema(estado,m1,m2): #e momentos linear e angular
    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2=estado[:8] #tira o tempo do estado para que não tena uma variável inútil sendo colocada aqui

    r1=np.array([x1,y1])
    r2=np.array([x2,y2])

    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)

    cinetica=((vx1**2+vy1**2)*m1/2+(vx2**2+vy2**2)*m2/2)
    
    potencial=-G*(m1*m2/r12)
    
    #r_min.append(r12)

    return cinetica+potencial


#retorna um valor aleatório de um parâmetro (multiplica por algum valor entre 0 e 1), podendo ser negativo ou positivo
def rd(a):
    return float(a)*(2*np.random.rand()-1)


def rotacionaVetor(v,eixo,angulo):
    pt1=v*np.cos(angulo)
    pt2=np.cross(eixo,v)*np.sin(angulo)
    pt3=eixo*np.dot(eixo,v)*(1-np.cos(angulo))
    return pt1+pt2+pt3


#============================== VARIÁVEIS DA SIMULAÇÃO =================================
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsLua.txt'),"LUA")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsTerra.txt'),"TERRA")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsSol.txt'),"SOL")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsPlutao.txt'),"PLUTAO")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsCaronte.txt'),"CARONTE")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsMarte.txt'),"MARTE")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsFobos.txt'),"FOBOS")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsJupiter.txt'),"JUPITER")
#salvarDadosHorizonsNPZ(extrair_horizons('simulacoesArtificiais/testeHorizonsIo.txt'),"IO")

estado0=np.array([
    -10, -5, .11, .5,
     10, -10, -.2, -.6
],dtype=float)

estadoTerraLua=np.array([ #tudo está no SI
    -1.049889067625132e11, -1.095550719362653e11, 2.098005027956806e4, -2.076878134531754e4,
    -1.048920118008394e11, -1.099460696073275e11, 2.191567044477730e4, -2.051617942441162e4
])

estadoPluaoCaronte=np.array([
    2.927139631843598e12, -4.417507817704321e12, 4.669582534338675e3, 1.786575359875376e3,
    2.927148252819081e12, -4.417492764609848e12, 4.800949760768632e3, 1.818717640016949e3
])

estadoMarteFobos=np.array([
    2.079073439334056e11, 8.041881632569720e9, -8.723202904732225e1, 2.628131372371663e4,
    2.079095157887138e11, 8.033085283416674e9, 1.792962620614684e3, 2.692736038731082e4
])

estadoJupiterIo=np.array([
    -3.851347581041645e11, 6.850342840540687e11, -1.154481492425305e4, -5.785542092116165e3,
    -3.848651961987973e11, 6.853579292446296e11, -2.492401912146892e4, 5.256511031361246e3
])

mTerra=5.97219e24
mLua=7.349e22
mSol=1988410e24
mPlutao=1.307e22
mCaronte=1.586e21
mMarte=6.4171e23
mFobos=1.08e16
mJupiter=18.9819e26
mIo=8.93e22

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

epsilon=1e-12 #para ser uma simulação fisicamente acurada/real essa variável tem que ser praticamente ZERO
steps=200000
N=0#VALORES PADRÃO
dt=10#VALORES PADRÃO
m1simulacao=0
m2simulacao=0

num_simul=2
j=0

#OBS: regra de ouro é manter PELO MENOS 100 steps por órbita, não menos pois senão pode haver proximidade excessiva dependendo do par de corpos em questão

#flag = 0  -->  simulador ALEATÓRIO
#flag = 1  -->  simulador HORIZONS TERRA-LUA - obliquidade 23.4º e º
#flag = 2  -->  simulador HORIZONS PLUTÃO-CARONTE - obliquidade 122.5º e 119º - (a componente Z da velocidade aqui é tão grande que impossibilita uma órbita 2D sem a considerar nos cálculos -sem a componente Z ~135 m/s, com Z ~223 m/s- os dados da HORIZONS deveriam ser rotacionados para encontrar a componente Z zerada e permitir uma simulação 2D - projeção da real órbita Caronte-Plutão)
#flag = 3  -->  simulador HORIZONS MARTE-PHOBOS - obliquidade 25.9º e 26º - (inclinação de Fobos é de ~25º em relação à eclíptica, então o gráfico da distância relativa é sem levar em consideração a amplitude de movimento do eixo Z)
#flag = 4  -->  simulador HORIZONS JÚPITER-IO - obliquidade 3.13º e 3.17º 
flagTipoDeSimulacao=4

if flagTipoDeSimulacao==0:
    dt=0.00025
    G=1

    while j<num_simul:
        print(j)
        x1,y1,vx1,vy1,\
        x2,y2,vx2,vy2=estado0

        #alternado as massas (mas mantendo a massa total = 1)
        m1simulacao=np.random.rand()
        m2simulacao=1-m1simulacao
        #alterando as posições e velocidades iniciais do sistema
        estado=np.array([rd(a) for a in[ #gera numeros aleatórios baseando-se nos valores origianais do estado0, sem alterá-los no molde
            x1,y1,vx1,vy1,\
            x2,y2,vx2,vy2
        ]])
        print("ESTADO0: ",estado0)
        print("MASSA ORIGINAL: ",m1simulacao,m2simulacao)
        print("\nESTADO GERADO: ",estado)
        print("\nMASSA ALTERADA: ",m1simulacao,m2simulacao,"\n")
        print("SOMA MASSAS:",m1simulacao+m2simulacao)
        #mantendo o momento linear do sistema, forçadamente, em ZERO
        r1=np.array([x1,y1])
        r2=np.array([x2,y2])

        v1=np.array([vx1,vy1])
        v2=np.array([vx2,vy2])

        P_total=m1simulacao*v1+m2simulacao*v2
        V_cm=P_total/(m1simulacao+m2simulacao)

        v1-=V_cm
        v2-=V_cm

        #colocando o centro de massa do sistema na origem da simulação, para facilitar o treinamento da rede (evita que ela tenha que aprender o DRIFT - o centro de massa estar em certa posição não afeta na evolução da simulação)
        R_cm=(m1simulacao*r1 + m2simulacao*r2)/(m1simulacao+m2simulacao)

        r1-=R_cm
        r2-=R_cm

        trajetoria=[]
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
            trajetoria.append(estadoComTempo)#aqui o trajetoria é uma lista de arrays
            tempoSimulacao.append(tAtual)

            energiaDoSistema.append(calculaEnergiaDoSistema(estado,m1simulacao,m2simulacao))
            
            #evolução do sistema
            estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao)

            #PARA EVITAR COLISÕES, QUE NÃO É O OBJETIVO DA REDE COMPREENDER, COLOCO ESSE GATILHO PARA EVITAR COLOCAR OS DADOS DA TRAJETÓRIA NO DATASET
            r12=np.sqrt((np.linalg.norm(np.array([estado[4],estado[5]])-np.array([estado[0],estado[1]])))**2 + epsilon**2)
            
            #progresso da simulação em porcentagem 
            if passoAtual % 10000 == 0 and passoAtual > 0:
                print(f"Passo {passoAtual}/{steps} ({100*passoAtual/steps:.1f}%)")
            if r12<0.2:
                print("COLISAO DETECTADA, CANCELANDO SIMULAÇÃO")
                print("\nPASSO \n",passoAtual)
                flag_colisao=1
                break
            
        #salva os dados da simulação se não houve colisão
        if(flag_colisao==0):  
            massas=[m1simulacao,m2simulacao]
            salvarEstadosNPZ(massas,trajetoria,tempoSimulacao,energiaDoSistema,momAng,momLin,r_min,j)
            j+=1


            trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos
            energiaDoSistema=np.array(energiaDoSistema)
            #print(trajetoria.shape)


            #plotando a imagem da trajetória
            x1=trajetoria[:,0]
            y1=trajetoria[:,1]

            x2=trajetoria[:,4]
            y2=trajetoria[:,5]


            #==============================GERAÇÃO DE PLOTS EM UMA MESMA IMAGEM==============================

            fig, axs = plt.subplots(4, 2, figsize=(12, 12))

            # ---------------- TRAJETÓRIA ----------------
            axs[0,0].plot(x1, y1, label='corpo 1')
            axs[0,0].plot(x2, y2, label='corpo 2')
            axs[0,0].scatter(x1[0], y1[0])
            axs[0,0].scatter(x2[0], y2[0])
            axs[0,0].set_title("Trajetória")
            axs[0,0].axis("equal")
            axs[0,0].grid()

            #  TRAJETÓRIA SALVA EM NPZ - apenas para verificar que 
            dados=[]
            dados=carregarEstadosNPZ(j-1)
            x1=dados[:,0]
            y1=dados[:,1]

            x2=dados[:,4]
            y2=dados[:,5]

            axs[3,0].plot(x1, y1, label='corpo 1')
            axs[3,0].plot(x2, y2, label='corpo 2')
            axs[3,0].scatter(x1[0], y1[0])
            axs[3,0].scatter(x2[0], y2[0])
            axs[3,0].set_title("Trajetória do arquivo NPZ")
            axs[3,0].axis("equal")
            axs[3,0].grid()

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
elif flagTipoDeSimulacao==1:
    N=120 #numero de subpassos
    dt=3600/N

    steps = 721
    G=6.6743e-11

    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2=estadoTerraLua


    m1simulacao=mTerra
    m2simulacao=mLua

    xt,yt,vxt,vyt=carregarDadosHorizonsNPZ("TERRA")
    xl,yl,vxl,vyl=carregarDadosHorizonsNPZ("LUA") 
elif flagTipoDeSimulacao==2:
    steps = 721
    G=6.6743e-11
    N=12
    dt=3600/N

    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2=estadoPluaoCaronte

    m1simulacao=mPlutao
    m2simulacao=mCaronte

    xt,yt,vxt,vyt=carregarDadosHorizonsNPZ("PLUTAO")
    xl,yl,vxl,vyl=carregarDadosHorizonsNPZ("CARONTE")    
elif flagTipoDeSimulacao==3:

    N=120 #passos entre uma verificação e outra - assim vai rodar mais passos sem ter que comparar os dados da HORIZONS
    dt=600/N
    steps = 721
    G=6.6743e-11

    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2=estadoMarteFobos

    m1simulacao=mMarte
    m2simulacao=mFobos

    xt,yt,vxt,vyt=carregarDadosHorizonsNPZ("MARTE")
    xl,yl,vxl,vyl=carregarDadosHorizonsNPZ("FOBOS")
elif flagTipoDeSimulacao==4:

    N=120 #passos entre uma verificação e outra - assim vai rodar mais passos sem ter que comparar os dados da HORIZONS
    dt=3600/N
    steps = 721
    G=6.6743e-11

    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2=estadoJupiterIo
    m1simulacao=mJupiter
    m2simulacao=mIo

    xt,yt,vxt,vyt=carregarDadosHorizonsNPZ("JUPITER")
    xl,yl,vxl,vyl=carregarDadosHorizonsNPZ("IO")

R_cm0=(m1simulacao*np.array([x1,y1])+m2simulacao*np.array([x2,y2]))/(m1simulacao+m2simulacao)
V_cm0=(m1simulacao*np.array([vx1,vy1])+m2simulacao*np.array([vx2,vy2]))/(m1simulacao+m2simulacao)

estado=np.array([
    x1-R_cm0[0],y1-R_cm0[1],vx1-V_cm0[0],vy1-V_cm0[1],
    x2-R_cm0[0],y2-R_cm0[1],vx2-V_cm0[0],vy2-V_cm0[1]
])

xt,yt,vxt,vyt=xt*1e3,yt*1e3,vxt*1e3,vyt*1e3
xl,yl,vxl,vyl=xl*1e3,yl*1e3,vxl*1e3,vyl*1e3

#aplicando a rotação de Rodrigues para que o plano orbital esteja no plano XY gerado pela simulação
#rRelRot=xt[0]-xl[0],yt[0]-yl[0],zt[0]-zl[0]

passoAtual=0
flag_colisao=0
tAtual=0.0

for i in range(steps):
    passoAtual+=1

    for _ in range(N):
        estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao)
    match flagTipoDeSimulacao:
        case 1: tAtual+=3600
        case 2: tAtual+=3600
        case 3: tAtual+=600
        case 4: tAtual+=3600
        case _: tAtual+=1

    estadoComTempo=np.append(estado.copy(),tAtual)
    trajetoria.append(estadoComTempo)
    tempoSimulacao.append(tAtual)

    energiaDoSistema.append(calculaEnergiaDoSistema(estado,m1simulacao,m2simulacao))
    #lembrando que o referencial do HORIZONS é diferente do referencial usado na simulação, pois no horizons usa-se o baricentro do sistema solar
    R_cm_HOR=(m1simulacao*np.array([xt[i],yt[i]])+m2simulacao*np.array([xl[i],yl[i]]))/(m1simulacao+m2simulacao)
    V_cm_HOR=(m1simulacao*np.array([vxt[i],vyt[i]])+m2simulacao*np.array([vxl[i],vyl[i]]))/(m1simulacao+m2simulacao)
    estadoHorizons=[
        xt[i]-R_cm_HOR[0],yt[i]-R_cm_HOR[1],vxt[i]-V_cm_HOR[0],vyt[i]-V_cm_HOR[1],
        xl[i]-R_cm_HOR[0],yl[i]-R_cm_HOR[1],vxl[i]-V_cm_HOR[0],vyl[i]-V_cm_HOR[1]
    ]

    energiaHorizons.append(calculaEnergiaDoSistema(estadoHorizons,m1simulacao,m2simulacao))
    
    diferencaEnergiaHorizonsXSimulacao.append(energiaHorizons[i]-energiaDoSistema[i])

    r1 = estado[0:2]
    v1 = estado[2:4]
    r2 = estado[4:6]
    v2 = estado[6:8]
    r_dinamicoSistema.append(np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2))

    momLinSistema.append(np.linalg.norm(m1simulacao*v1 + m2simulacao*v2))
    momAngSistema.append(m1simulacao*(r1[0]*v1[1] - r1[1]*v1[0]) + m2simulacao*(r2[0]*v2[1] - r2[1]*v2[0]))

    #dados do horizons
    x1h,y1h,vx1h,vy1h,x2h,y2h,vx2h,vy2h=estadoHorizons

    r1h=np.array([x1h,y1h])
    r2h=np.array([x2h,y2h])
    r_dinamicoHorizons.append(np.sqrt((np.linalg.norm(r2h-r1h))**2 + epsilon**2))
    v1h=np.array([vx1h,vy1h])
    v2h=np.array([vx2h,vy2h])

    momLinHorizons.append(np.linalg.norm(m1simulacao*v1h + m2simulacao*v2h))
    momAngHorizons.append(m1simulacao*(r1h[0]*v1h[1] - r1h[1]*v1h[0]) + m2simulacao*(r2h[0]*v2h[1] - r2h[1]*v2h[0]))

    #append das diferenças
    diferencaMomLinHorizonsXSimulacao.append(momLinHorizons[i]-momLinSistema[i])
    diferencaMomAngHorizonsXSimulacao.append(momAngHorizons[i]-momAngSistema[i])

    diferencaDistanciaHorizonsXSimulacao.append(r_dinamicoHorizons[i]-r_dinamicoSistema[i])
    
    # diferença de posição relativa
    r_rel_sim=estado[4:6]-estado[0:2]  # simulador
    r_rel_hor=np.array([xl[i]-xt[i],yl[i]-yt[i]])  #HORIZONS

    erro_posicao.append(np.linalg.norm(r_rel_sim-r_rel_hor))
    erro_relativo_posicao=100*np.array(erro_posicao)/np.array(r_dinamicoSistema)

    #evolução do sistema
    #estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao)

    #progresso da simulação em porcentagem 
    if passoAtual % 10000 == 0 and passoAtual > 0:
        print(f"Passo {passoAtual}/{steps} ({100*passoAtual/steps:.1f}%)")
#==========================================GERAÇÃO DE PLOTS EM UMA MESMA IMAGEM==========================================
trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos

#plotando a imagem da trajetória
x1=trajetoria[:,0]
y1=trajetoria[:,1]

x2=trajetoria[:,4]
y2=trajetoria[:,5]

fig, axs = plt.subplots(4, 2, figsize=(12, 12))

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

# ---------------- Diferença no Momento Linear ----------------
axs[1,0].plot(diferencaMomLinHorizonsXSimulacao)
axs[1,0].set_title("Diferença no Momento Linear")
axs[1,0].grid()

# ---------------- Momento Angular ----------------
axs[1,1].plot(diferencaMomAngHorizonsXSimulacao)
axs[1,1].set_title("Diferença no Momento Angular")
axs[1,1].grid()

# ---------------- Energia ----------------
axs[2,0].plot(energiaDoSistema)
axs[2,0].set_title("Energia Total")
axs[2,0].ticklabel_format(useOffset=False, style='sci', axis='y')
axs[2,0].grid()

# ---------------- Energia dos dados HORIZONS----------------
r_sim_km = np.array(r_dinamicoSistema) / 1e3
r_hor_km = np.array(r_dinamicoHorizons) / 1e3

axs[2,1].plot(r_sim_km, label='Simulador', lw=1)
axs[2,1].plot(r_hor_km, label='HORIZONS (2D)', lw=1, alpha=0.7)
axs[2,1].set_title("Distância relativa (km)")
axs[2,1].legend(fontsize=7)
axs[2,1].grid()

# ---------------- Erro relativo SIMULADOR----------------
E0 = energiaDoSistema[0]
erro_relativo = (energiaDoSistema - E0)/abs(E0)
axs[3,0].plot(erro_relativo)
axs[3,0].set_title("Erro Relativo da Energia do SIMULADOR")
axs[3,0].grid()

# ---------------- Diferença absoluta de distância ----------------
axs[3,1].plot(r_hor_km - r_sim_km)
axs[3,1].set_title("Diferença de distância HORIZONS - SIMULADOR (km)")
axs[3,1].grid()

plt.tight_layout()
plt.show()

# O simulador de 2 corpos com integrador Yoshida 4ª ordem reproduz a dinâmica real 
# Terra-Lua com erro posicional relativo abaixo de 1% nos primeiros 10 dias, crescendo 
# para ~13% ao final de 30 dias (efeito de maré, perturbação solar e de outros planetas, pequenos efeitos relativisticos).
# O crescimento do erro é atribuído às perturbações gravitacionais do
# Sol e demais corpos, ausentes no modelo de 2 corpos isolados, e não a imprecisões numéricas do integrador 
# — confirmado pelo erro relativo de energia na ordem de 10e-10.
