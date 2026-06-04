import numpy as np, matplotlib.pyplot as plt


'''def estadoAtual(estado,m1,m2,m3):
    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2,\
    x3,y3,vx3,vy3,t=estado #pareando as variáveis com seus respectivos valores, onde x1=estado[0]
    #vetores posição
    r1=np.array([x1,y1])
    r2=np.array([x2,y2])
    r3=np.array([x3,y3])
    #distâncias
    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)#traz a norma da matriz/vetor (distância entre os dois pontos/vetores posição) e o sqrt com a soma do 1e-6 garante que a distância não será 0 para evitar erro em divisão
    r13=np.sqrt((np.linalg.norm(r3-r1))**2 + epsilon**2)
    r23=np.sqrt((np.linalg.norm(r3-r2))**2 + epsilon**2)

    #acelerações
    a1=G*m2*(r2-r1)/r12**3 + G*m3*(r3-r1)/r13**3
    a2=G*m1*(r1-r2)/r12**3 + G*m3*(r3-r2)/r23**3
    a3=G*m1*(r1-r3)/r13**3 + G*m2*(r2-r3)/r23**3

    #retornando as derivadas masi o dt/dt=1
    return np.array([vx1,vy1,a1[0],a1[1],
                     vx2,vy2,a2[0],a2[1],
                     vx3,vy3,a3[0],a3[1],
                     1])'''

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

def salvarEstadosNPZ(massas,trajetoria,tempo,energia,momAng,momLin,r_min,aceleracoes,nome):
    #convertendo para array caso ainda não seja
    massas=np.array(massas)
    trajetoria=np.array(trajetoria)
    tempo=np.array(tempo)
    energia=np.array(energia)
    momAng=np.array(momAng)
    momLin=np.array(momLin)
    r_min=np.array(r_min)
    aceleracoes=np.array(aceleracoes)

    #para facilitar no treinamento da rede, salvo todos os arrays em um arquivo compactado
    np.savez_compressed(f"simulacoesArtificiais/simulacao3C{nome}.npz",
                        massas=massas,trajetoria=trajetoria,tempo=tempo,energia=energia,momAng=momAng,momLin=momLin,r_min=r_min,aceleracoes=aceleracoes)
    print(f"Arquivo salvo. Tamanho aproximado: {trajetoria.nbytes/1e6:.1f} MB")


#função para carregar os dados depois
def carregarEstadosNPZ(nome):
    dados=np.load(f"simulacoesArtificiais/simulacao3C{nome}.npz")
    return dados['trajetoria']

'''def salvarEstadosNPZ(massas,trajetoria,tempo,energia,momAng,momLin,r_min,nome):
    #convertendo para array caso ainda não seja
    massas=np.array(massas)
    trajetoria=np.array(trajetoria)
    tempo=np.array(tempo)
    energia=np.array(energia)
    momAng=np.array(momAng)
    momLin=np.array(momLin)
    r_min=np.array(r_min)

    #para facilitar no treinamento da rede, salvo todos os arrays em um arquivo compactado
    np.savez_compressed(f"simulacoesArtificiais/simulacao{nome}.npz",
                        massas=massas,trajetoria=trajetoria,tempo=tempo,energia=energia,momAng=momAng,momLin=momLin,r_min=r_min)
    print(f"Arquivo salvo. Tamanho aproximado: {trajetoria.nbytes/1e6:.1f} MB")


#função para carregar os dados depois
def carregarEstadosNPZ(nome):
    dados=np.load(f"simulacoesArtificiais/simulacao{nome}.npz")
    return dados['trajetoria']
'''

'''def rk4(estado,dt,m1,m2,m3):#por mais que seja muito preciso LOCALMENTE, excelente para curto prazo, mas não é SIMPLÉTICO (não preserva a geometria do espaço de fases), não conserva a energia do sistema hamiltoniano, por isso aplico LEAPFROG abaixo
    estado=estado.copy()  

    k1=estadoAtual(estado,m1,m2,m3)
    k2=estadoAtual(estado+0.5*dt*k1,m1,m2,m3)
    k3=estadoAtual(estado+0.5*dt*k2,m1,m2,m3)
    k4=estadoAtual(estado+dt*k3,m1,m2,m3)

    return estado + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


#o método do leapfrog vai encontrar o meio passo de velocidade, depois o passo inteiro de posição para então atualizar a aceleração com a nova posição (preciso de uma função apenas para atualizar a aceleração) e finalizar a velocidade
def leapfrog(estado,dt,m1,m2,m3):#alterna posição em tempo inteiro e velocidade em meio passo de tempo
    estado=estado.copy()
       
    #separando posições e velocidades
    r1=estado[0:2]
    v1=estado[2:4]
    r2=estado[4:6]
    v2=estado[6:8]
    r3=estado[8:10]
    v3=estado[10:12]

    #aceleração inicial
    a1,a2,a3=atualizaAceleracoes_estado(estado,m1,m2,m3)

    #meio passo (velocidade)
    v1_meio=v1+dt*a1/2
    v2_meio=v2+dt*a2/2
    v3_meio=v3+dt*a3/2

    #passo completo da posição
    r1_novo=r1+dt*v1_meio
    r2_novo=r2+dt*v2_meio
    r3_novo=r3+dt*v3_meio

    estadoTemporario=np.array([#para poder calcular a nova aceleração
        r1_novo[0],r1_novo[1],v1_meio[0],v1_meio[1],
        r2_novo[0],r2_novo[1],v2_meio[0],v2_meio[1],
        r3_novo[0],r3_novo[1],v3_meio[0],v3_meio[1]
    ])

    a1_nova,a2_nova,a3_nova=atualizaAceleracoes_estado(estadoTemporario,m1,m2,m3)

    #finalizando a velocidade
    v1_nova=v1_meio+dt*a1_nova/2
    v2_nova=v2_meio+dt*a2_nova/2
    v3_nova=v3_meio+dt*a3_nova/2

    return np.array([
        r1_novo[0], r1_novo[1], v1_nova[0], v1_nova[1],
        r2_novo[0], r2_novo[1], v2_nova[0], v2_nova[1],
        r3_novo[0], r3_novo[1], v3_nova[0], v3_nova[1]
    ])'''


def yoshida4ordem(estado,dt,m1,m2,m3):#utiliza algumas vezes o velocity-verlet
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
    r3=estado[8:10]
    v3=estado[10:12]

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


#VERSÃO QUE NÃO CHAMA O NP.LINALOG.NORM, SÓ CONTENDO POUCOS SQRT E MULTIPLICAÇÕES, SEM DIVISÕES (que são mais lentas que as multiplicações)
def atualizaAceleracoes_posicoes(r1,r2,r3,m1,m2,m3):
    #vetor posição
    pr12=r2-r1
    pr13=r3-r1
    pr23=r3-r2
    #radicando
    rr12=pr12[0]**2 + pr12[1]**2 + epsilon**2
    rr13=pr13[0]**2 + pr13[1]**2 + epsilon**2
    rr23=pr23[0]**2 + pr23[1]**2 + epsilon**2
    #fazendo o inverso para poupar algumas divisões

    inv_r12=1/(rr12*np.sqrt(rr12))
    inv_r13=1/(rr13*np.sqrt(rr13))
    inv_r23=1/(rr23*np.sqrt(rr23))

    a1=G*(m2*pr12*inv_r12 + m3*pr13*inv_r13)
    a2=G*(-m1*pr12*inv_r12 + m3*pr23*inv_r23)
    a3=G*(-m1*pr13*inv_r13 - m2*pr23*inv_r23)

    return a1,a2,a3


def atualizaAceleracoes_estado(estado,m1,m2,m3):
    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2,\
    x3,y3,vx3,vy3=estado

    r1=np.array([x1,y1])
    r2=np.array([x2,y2])
    r3=np.array([x3,y3])
    
    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)
    r13=np.sqrt((np.linalg.norm(r3-r1))**2 + epsilon**2)
    r23=np.sqrt((np.linalg.norm(r3-r2))**2 + epsilon**2)
    
    a1=G*m2*(r2-r1)/r12**3 + G*m3*(r3-r1)/r13**3
    a2=G*m1*(r1-r2)/r12**3 + G*m3*(r3-r2)/r23**3
    a3=G*m1*(r1-r3)/r13**3 + G*m2*(r2-r3)/r23**3

    return a1,a2,a3


def calculaEnergiaDoSistema(estado,m1,m2,m3): #e momentos linear e angular
    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2,\
    x3,y3,vx3,vy3=estado[:12] #tira o tempo do estado para que não tena uma variável inútil sendo colocada aqui

    r1=np.array([x1,y1])
    r2=np.array([x2,y2])
    r3=np.array([x3,y3])

    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)
    r13=np.sqrt((np.linalg.norm(r3-r1))**2 + epsilon**2)
    r23=np.sqrt((np.linalg.norm(r3-r2))**2 + epsilon**2)

    #v1=np.array([vx1,vy1])
    #v2=np.array([vx2,vy2])
    #v3=np.array([vx3,vy3])

    #massas=np.array([m1,m2,m3])

    #momLin.append(np.linalg.norm(m1*v1 + m2*v2 + m3*v3))

    #momAng.append(m1*(r1[0]*v1[1] - r1[1]*v1[0]) + m2*(r2[0]*v2[1] - r2[1]*v2[0]) + m3*(r3[0]*v3[1] - r3[1]*v3[0]))

    cinetica=((vx1**2+vy1**2)*m1/2+(vx2**2+vy2**2)*m2/2+(vx3**2+vy3**2)*m3/2)
    
    potencial=-G*(m1*m2/r12+m1*m3/r13+m2*m3/r23)
    
    #r_min.append(min(r12,r13,r23))

    return cinetica+potencial


#retorna um valor aleatório de um parâmetro (multiplica por algum valor entre 0 e 1), podendo ser negativo ou positivo
def rd(a):
    return float(a)*(2*np.random.rand()-1)


#a rotação de Rodrigues não funcionará 100% com os 3 corpos pois os planos prbitais não são coincidentes muitas vezes
#SOLUÇÃO: aplicar mais de uma vez a rotação de Rodrigues --> aplico no eixo dominante (Sol-Terra) e depois aplico a mesma rotaç~onos vetores posição e velocidade da Lua, assim todos os corpos estarão no mesmo plano prbital --: CENÁRIO IDEALIZADO, APENAS PARA APRENDIZAGEM
def rotacionaVetor(v,eixo,angulo):
    pt1=v*np.cos(angulo)
    pt2=np.cross(eixo,v)*np.sin(angulo)
    pt3=eixo*np.dot(eixo,v)*(1-np.cos(angulo))
    return pt1+pt2+pt3

def calculaMomLin(m1,m2,m3,v1,v2,v3):
    return np.linalg.norm(m1*v1 + m2*v2 + m3*v3)

def calculaMomAng(m1,m2,m3,r1,r2,r3,v1,v2,v3):
    #return m1*(r1[0]*v1[1] - r1[1]*v1[0]) + m2*(r2[0]*v2[1] - r2[1]*v2[0])
    return m1*(r1[0]*v1[1] - r1[1]*v1[0]) + m2*(r2[0]*v2[1] - r2[1]*v2[0]) + m3*(r3[0]*v3[1] - r3[1]*v3[0])

def validarSimulador(trajetoria,massas,dt,calcularEnergia,calcularMomAng,calcularMomLin,integrador):
    massas=np.array(massas)
    trajetoria=np.array(trajetoria)


    #======calculo das invariantes do sistema======
    energias,momLin,momAng=[],[],[]
    for estado in trajetoria[:, :-1]: #aqui o [:, :-1] significa que vai pegar todas as linhas individualmente mas MENOS UMA coluna, que é a final (onde há o tempo, inútil nestes cálculos daqui)
        r1=estado[0:2]
        v1=estado[2:4]
        r2=estado[4:6]
        v2=estado[6:8]
        r3=estado[8:10]
        v3=estado[10:12]
        energias.append(calcularEnergia(estado,massas[0],massas[1],massas[2]))
        momLin.append(calcularMomLin(massas[0],massas[1],massas[2],v1,v2,v3))
        momAng.append(calcularMomAng(massas[0],massas[1],massas[2],r1,r2,r3,v1,v2,v3))
    
    energias=np.array(energias)
    momLin=np.array(momLin)
    momAng=np.array(momAng)

    desvioEnerg=(energias.max()-energias.min())/abs(energias.mean())
    #aqui precisei colocar baseando-se na dimensão dos vetores dos momentos para evitar runtime error na simulação 2D, e como é uma função genérica para os 4 simuladores foi necessário manter como condição mesmo
    escala = np.abs(massas[0] * np.linalg.norm(trajetoria[0, 2:4]))
    #HOUSTON, WE HAVE A PROBLEM
    if momAng.ndim==1 and momLin.ndim==1:
        desvioAng=np.max(np.abs(momAng-momAng[0]))/np.abs(momAng[0])
        desvioLin=np.max(np.abs(momLin))/escala#np.abs(momLin[0])
    else:
        desvioAng=np.max(np.linalg.norm(momAng-momAng[0],axis=-1))/np.linalg.norm(momAng[0]) #lembrando que np.linalg.norm retorna a magnitude, e nesse caso é a magnitude da diferença entre momAng inicial e o atual
        desvioLin=np.max(np.linalg.norm(momLin,axis=-1))/np.linalg.norm(momLin[0])
   

    #======reversibilidade======
    estado=trajetoria[0, :-1].copy() #copia a primeira linha sem o tempo
    steps=50000#len(trajetoria)
    for _ in range(steps):
        estado=integrador(estado,dt,massas[0],massas[1],massas[2])
    for _ in range(steps):
        estado=integrador(estado,-dt,massas[0],massas[1],massas[2])
    erroReversao=np.linalg.norm(estado-trajetoria[0, :-1])/np.linalg.norm(trajetoria[0, :-1])

    return [desvioEnerg,desvioLin,desvioAng,erroReversao]




#============================== VARIÁVEIS DA SIMULAÇÃO =================================
"""estado0=np.array([
    -10, -5, .11, 0.05,
     10, -10, -.2, -0.06,
     5, 10, -0.055, 0
],dtype=float)"""

estado0=np.array([
    -10, -5, .11, .5,
     10, -10, -.2, -.6,
     5, 10, -.55, 0
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

epsilon=1e-12 #para ser uma simulação fisicamente acurada/real essa variável tem que ser praticamente ZERO
steps=200000
N=0#VALORES PADRÃO
dt=10#VALORES PADRÃO
m1simulacao=0
m2simulacao=0

num_simul=5
j=0

#OBS: regra de ouro é manter PELO MENOS 100 steps por órbita, não menos pois senão pode haver proximidade excessiva dependendo do par de corpos em questão

#flag = 0  -->  simulador ALEATÓRIO
#flag = 1  -->  simulador HORIZONS TERRA-LUA-SOL
#flag = 2  -->  simulador HORIZONS 
#flag = 3  -->  simulador HORIZONS 
#flag = 4  -->  simulador HORIZONS 
flagTipoDeSimulacao=1

if flagTipoDeSimulacao==0:
    dt=0.00025
    G=1

    while j<num_simul:
        
        rMomentaneo=[]
        momLin=[]
        momAng=[]
        aceleracoes=[]

        x1,y1,vx1,vy1,\
        x2,y2,vx2,vy2,\
        x3,y3,vx3,vy3=estado0

        #alternado as massas (mas mantendo a massa total = 1)
        m1simulacao=np.random.rand()
        resto=1-m1simulacao
        m2simulacao=resto*np.random.rand()
        m3simulacao=1-m1simulacao-m2simulacao

        #alterando as posições e velocidades iniciais do sistema
        estadoAleatorio=np.array([rd(a) for a in[ #gera numeros aleatórios baseando-se nos valores origianais do estado0, sem alterá-los no molde
            x1,y1,vx1,vy1,\
            x2,y2,vx2,vy2,\
            x3,y3,vx3,vy3
        ]])
        '''print("ESTADO0: ",estado0)
        print("MASSA ORIGINAL: ",m1simulacao,m2simulacao)
        print("\nESTADO GERADO: ",estado)
        print("\nMASSA ALTERADA: ",m1simulacao,m2simulacao,"\n")
        print("SOMA MASSAS:",m1simulacao+m2simulacao)'''
        x1,y1,vx1,vy1,x2,y2,vx2,vy2,x3,y3,vx3,vy3=estadoAleatorio

        #mantendo o momento linear do sistema, forçadamente, em ZERO        P_total=m1simulacao*v1+m2simulacao*v2+m3simulacao*v3       
        r1=np.array([x1,y1])
        r2=np.array([x2,y2])
        r3=np.array([x3,y3])

        v1=np.array([vx1,vy1])
        v2=np.array([vx2,vy2])
        v3=np.array([vx3,vy3])
        
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
            r1[0],r1[1],v1[0],v1[1],
            r2[0],r2[1],v2[0],v2[1],
            r3[0],r3[1],v3[0],v3[1]
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

            energiaDoSistema.append(calculaEnergiaDoSistema(estado,m1simulacao,m2simulacao,m3simulacao))
            '''a1,a2=atualizaAceleracoes_posicoes(r1,r2,m1simulacao,m2simulacao)
            aceleracoes.append(np.concatenate([a1,a2]))
            #evolução do sistema
            estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao)'''

            #PARA EVITAR COLISÕES, QUE NÃO É O OBJETIVO DA REDE COMPREENDER, COLOCO ESSE GATILHO PARA EVITAR COLOCAR OS DADOS DA TRAJETÓRIA NO DATASET
            r12=np.sqrt((np.linalg.norm(np.array([estado[4],estado[5]])-np.array([estado[0],estado[1]])))**2 + epsilon**2)
            r13=np.sqrt((np.linalg.norm(np.array([estado[8],estado[9]])-np.array([estado[0],estado[1]])))**2 + epsilon**2)
            r23=np.sqrt((np.linalg.norm(np.array([estado[8],estado[9]])-np.array([estado[4],estado[5]])))**2 + epsilon**2)#otimizar isso aqui ****************************
        
            r1 = estado[0:2]
            v1 = estado[2:4]
            r2 = estado[4:6]
            v2 = estado[6:8]
            r3 = estado[8:10]
            v3 = estado[10:12]

            a1,a2,a3=atualizaAceleracoes_posicoes(r1,r2,r3,m1simulacao,m2simulacao,m3simulacao)
            aceleracoes.append(np.concatenate([a1,a2,a3]))
            #evolução do sistema
            estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao,m3simulacao)

            momLin.append(calculaMomLin(m1simulacao,m2simulacao,m3simulacao,v1,v2,v3))
            momAng.append(calculaMomAng(m1simulacao,m2simulacao,m3simulacao,r1,r2,r3,v1,v2,v3))

            #a distância momentânea nao se aplica mais dessa forma, teria que ter soma da distância entre os corpos pelo menos para fazer sentido
            rMomentaneo.append(np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)+np.sqrt((np.linalg.norm(r3-r1))**2 + epsilon**2)+np.sqrt((np.linalg.norm(r2-r3))**2 + epsilon**2))

            #progresso da simulação em porcentagem 
            if passoAtual % 10000 == 0 and passoAtual > 0:
                print(f"Passo {passoAtual}/{steps} ({100*passoAtual/steps:.1f}%)")
            if min(r12,r13,r23)<0.2:
                print("COLISAO DETECTADA, CANCELANDO SIMULAÇÃO\n\n")
                print("\nPASSO \n",passoAtual)
                flag_colisao=1
                break
            
        #salva os dados da simulação se não houve colisão
        if(flag_colisao==0):  
            massas=[m1simulacao,m2simulacao,m3simulacao]
            salvarEstadosNPZ(massas,trajetoria,tempoSimulacao,energiaDoSistema,momAng,momLin,rMomentaneo,aceleracoes,j)
            j+=1


            trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos
            energiaDoSistema=np.array(energiaDoSistema)
            #print(trajetoria.shape)

            #==============VALIDANDO O SIMULADOR#==============
            resultados=validarSimulador(trajetoria,massas,dt,calculaEnergiaDoSistema,calculaMomAng,calculaMomLin,yoshida4ordem)
            #resultados = [desvioEnerg,desvioLin,desvioAng,erroReversao]
            print(f"VALORES DA VALIDAÇÃO:\nDesvio da energia: {resultados[0]}\nDesvio do momento linear: {resultados[1]}\nDesvio do momento angular: {resultados[2]}\nErro da reversão: {resultados[3]:.2e}")


            #plotando a imagem da trajetória
            x1=trajetoria[:,0]
            y1=trajetoria[:,1]

            x2=trajetoria[:,4]
            y2=trajetoria[:,5]

            x3=trajetoria[:,8]
            y3=trajetoria[:,9]


            #==============================GERAÇÃO DE PLOTS EM UMA MESMA IMAGEM==============================

            fig, axs = plt.subplots(4, 2, figsize=(12, 12))

            # ---------------- TRAJETÓRIA ----------------
            axs[0,0].plot(x1, y1, label='corpo 1')
            axs[0,0].plot(x2, y2, label='corpo 2')
            axs[0,0].plot(x3, y3, label='corpo 3')
            axs[0,0].scatter(x1[0], y1[0])
            axs[0,0].scatter(x2[0], y2[0])
            axs[0,0].scatter(x3[0], y3[0])
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

            x3=dados[:,8]
            y3=dados[:,9]

            axs[3,0].plot(x1, y1, label='corpo 1')
            axs[3,0].plot(x2, y2, label='corpo 2')
            axs[3,0].plot(x3, y3, label='corpo 3')
            axs[3,0].scatter(x1[0], y1[0])
            axs[3,0].scatter(x2[0], y2[0])
            axs[3,0].scatter(x3[0], y3[0])
            axs[3,0].set_title("Trajetória do arquivo NPZ")
            axs[3,0].axis("equal")
            axs[3,0].grid()

            # ---------------- r_min ----------------
            axs[0,1].plot(rMomentaneo)
            axs[0,1].set_title("Distância entre os corpos")
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
            print("Órbita ligada" if E0<0 else "Hiperbólica - não ligado")
            erro_relativo = (energiaDoSistema - E0)/abs(E0)
            axs[2,1].plot(erro_relativo)
            axs[2,1].set_title("Erro Relativo da Energia")
            axs[2,1].grid()

            plt.tight_layout()
            plt.show()
else: 
    if flagTipoDeSimulacao==1:
        '''x1,y1,vx1,vy1,\
        x2,y2,vx2,vy2=estadoTerraLua'''

        N=120 #numero de subpassos
        dt=3600/N

        steps = 721
        G=6.6743e-11

        


        m1simulacao=mTerra
        m2simulacao=mLua
        m3simulacao=mSol
        xs,ys,zs,vxs,vys,vzs=carregarDadosHorizonsNPZ("SOL")
        xt,yt,zt,vxt,vyt,vzt=carregarDadosHorizonsNPZ("TERRA")
        xl,yl,zl,vxl,vyl,vzl=carregarDadosHorizonsNPZ("LUA")
         

        #estadoTerraLua=[xt[0],yt[0],zt[0],vxt[0],vyt[0],vzt[0],xl[0],yl[0],zl[0],vxl[0],vyl[0],vzl[0]]
    
    elif flagTipoDeSimulacao==2:
        steps = 721
        G=6.6743e-11
        N=12
        dt=3600/N

        '''x1,y1,vx1,vy1,\
        x2,y2,vx2,vy2=estadoPluaoCaronte'''

        m1simulacao=mPlutao
        m2simulacao=mCaronte
        m3simulacao=mSol
        xs,ys,zs,vxs,vys,vzs=carregarDadosHorizonsNPZ("SOL")
        xt,yt,zt,vxt,vyt,vzt=carregarDadosHorizonsNPZ("PLUTAO")
        xl,yl,zl,vxl,vyl,vzl=carregarDadosHorizonsNPZ("CARONTE")   

        #estadoPluaoCaronte=[xt[0],yt[0],zt[0],vxt[0],vyt[0],vzt[0],xl[0],yl[0],zl[0],vxl[0],vyl[0],vzl[0]] 

    elif flagTipoDeSimulacao==3:

        N=120 #passos entre uma verificação e outra - assim vai rodar mais passos sem ter que comparar os dados da HORIZONS
        dt=600/N
        steps = 721
        G=6.6743e-11

        '''x1,y1,vx1,vy1,\
        x2,y2,vx2,vy2=estadoMarteFobos'''

        m1simulacao=mMarte
        m2simulacao=mFobos
        m3simulacao=mSol
        xs,ys,zs,vxs,vys,vzs=carregarDadosHorizonsNPZ("SOL")
        xt,yt,zt,vxt,vyt,vzt=carregarDadosHorizonsNPZ("MARTE")
        xl,yl,zl,vxl,vyl,vzl=carregarDadosHorizonsNPZ("FOBOS")

        #estadoMarteFobos=[xt[0],yt[0],zt[0],vxt[0],vyt[0],vzt[0],xl[0],yl[0],zl[0],vxl[0],vyl[0],vzl[0]] 
    
    elif flagTipoDeSimulacao==4:

        N=120 #passos entre uma verificação e outra - assim vai rodar mais passos sem ter que comparar os dados da HORIZONS
        dt=3600/N
        steps = 721
        G=6.6743e-11

        '''x1,y1,vx1,vy1,\
        x2,y2,vx2,vy2=estadoJupiterIo'''
        m1simulacao=mJupiter
        m2simulacao=mIo
        m3simulacao=mSol
        xs,ys,zs,vxs,vys,vzs=carregarDadosHorizonsNPZ("SOL")
        xt,yt,zt,vxt,vyt,vzt=carregarDadosHorizonsNPZ("JUPITER")
        xl,yl,zl,vxl,vyl,vzl=carregarDadosHorizonsNPZ("IO")

        #estadoJupiterIo=[xt[0],yt[0],zt[0],vxt[0],vyt[0],vzt[0],xl[0],yl[0],zl[0],vxl[0],vyl[0],vzl[0]] 


    xt,yt,zt,vxt,vyt,vzt=xt*1e3,yt*1e3,zt*1e3,vxt*1e3,vyt*1e3,vzt*1e3
    xl,yl,zl,vxl,vyl,vzl=xl*1e3,yl*1e3,zl*1e3,vxl*1e3,vyl*1e3,vzl*1e3
    xs,ys,zs,vxs,vys,vzs=xs*1e3,ys*1e3,zs*1e3,vxs*1e3,vys*1e3,vzs*1e3

    #aplicando a rotação de Rodrigues para que o plano orbital esteja no plano XY gerado pela simulação
    rRelRot=xt[0]-xs[0],yt[0]-ys[0],zt[0]-zs[0]
    vRelRot=vxt[0]-vxs[0],vyt[0]-vys[0],vzt[0]-vzs[0]

    L=np.cross(rRelRot,vRelRot)

    zUnitario=np.array([0,0,1])
    LUnitario=L/np.linalg.norm(L)

    eixo=np.cross(LUnitario,zUnitario)
    eixo=eixo/np.linalg.norm(eixo)

    angulo=np.arccos(np.dot(LUnitario,zUnitario))
    

    #rotacionando com método de Rodrigues
    for i in range(len(xt)):
        rT=np.array([xt[i],yt[i],zt[i]])
        rL=np.array([xl[i],yl[i],zl[i]])
        rS=np.array([xs[i],ys[i],zs[i]])
        vT=np.array([vxt[i],vyt[i],vzt[i]])
        vL=np.array([vxl[i],vyl[i],vzl[i]])
        vS=np.array([vxs[i],vys[i],vzs[i]])
        
        '''
        def rotacionaVetor(v,eixo,angulo):
        pt1=v*np.cos(angulo)
        pt2=np.cross(eixo,v)*np.sin(angulo)
        pt3=eixo*np.dot(eixo,v)*(1-np.cos(angulo))
        return pt1+pt2+pt3
        '''


        rT_rot=rotacionaVetor(rT,eixo,angulo)
        rL_rot=rotacionaVetor(rL,eixo,angulo)
        rS_rot=rotacionaVetor(rS,eixo,angulo)
        vT_rot=rotacionaVetor(vT,eixo,angulo)
        vL_rot=rotacionaVetor(vL,eixo,angulo)
        vS_rot=rotacionaVetor(vS,eixo,angulo)
        
        
        xt[i],yt[i]=rT_rot[0],rT_rot[1]
        xl[i],yl[i]=rL_rot[0],rL_rot[1]
        xs[i],ys[i]=rS_rot[0],rS_rot[1]
        vxt[i],vyt[i]=vT_rot[0],vT_rot[1]
        vxl[i],vyl[i]=vL_rot[0],vL_rot[1]
        vxs[i],vys[i]=vS_rot[0],vS_rot[1]


    x1,y1,vx1,vy1=xt[0],yt[0],vxt[0],vyt[0]
    x2,y2,vx2,vy2=xl[0],yl[0],vxl[0],vyl[0]#isso substituiria a declaração dos valores de X e Y em estadoTerraLua... burrice minha, simplesmente
    x3,y3,vx3,vy3=xs[0],ys[0],vxs[0],vys[0]

    R_cm0=(m1simulacao*np.array([x1,y1])+m2simulacao*np.array([x2,y2])+m3simulacao*np.array([x3,y3]))/(m1simulacao+m2simulacao+m3simulacao)
    V_cm0=(m1simulacao*np.array([vx1,vy1])+m2simulacao*np.array([vx2,vy2])+m3simulacao*np.array([vx3,vy3]))/(m1simulacao+m2simulacao+m3simulacao)

    estado=np.array([
        x1-R_cm0[0],y1-R_cm0[1],vx1-V_cm0[0],vy1-V_cm0[1],
        x2-R_cm0[0],y2-R_cm0[1],vx2-V_cm0[0],vy2-V_cm0[1],
        x3-R_cm0[0],y3-R_cm0[1],vx3-V_cm0[0],vy3-V_cm0[1]
    ])




    passoAtual=0
    flag_colisao=0
    tAtual=0.0

    for i in range(steps):
        passoAtual+=1

        '''for _ in range(N):
            estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao)
        match flagTipoDeSimulacao:
            case 1: tAtual+=3600
            case 2: tAtual+=3600
            case 3: tAtual+=600
            case 4: tAtual+=3600
            case _: tAtual+=1'''

        estadoComTempo=np.append(estado.copy(),tAtual)
        trajetoria.append(estadoComTempo)
        tempoSimulacao.append(tAtual)

        energiaDoSistema.append(calculaEnergiaDoSistema(estado,m1simulacao,m2simulacao,m3simulacao))
        #lembrando que o referencial do HORIZONS é diferente do referencial usado na simulação, pois no horizons usa-se o baricentro do sistema solar
        R_cm_HOR=(m1simulacao*np.array([xt[i],yt[i]])+m2simulacao*np.array([xl[i],yl[i]])+m3simulacao*np.array([xs[i],ys[i]]))/(m1simulacao+m2simulacao+m3simulacao)
        V_cm_HOR=(m1simulacao*np.array([vxt[i],vyt[i]])+m2simulacao*np.array([vxl[i],vyl[i]])+m3simulacao*np.array([vxs[i],vys[i]]))/(m1simulacao+m2simulacao+m3simulacao)
        estadoHorizons=[
            xt[i]-R_cm_HOR[0],yt[i]-R_cm_HOR[1],vxt[i]-V_cm_HOR[0],vyt[i]-V_cm_HOR[1],
            xl[i]-R_cm_HOR[0],yl[i]-R_cm_HOR[1],vxl[i]-V_cm_HOR[0],vyl[i]-V_cm_HOR[1],
            xs[i]-R_cm_HOR[0],ys[i]-R_cm_HOR[1],vxs[i]-V_cm_HOR[0],vys[i]-V_cm_HOR[1]
        ]

        energiaHorizons.append(calculaEnergiaDoSistema(estadoHorizons,m1simulacao,m2simulacao,m3simulacao))
        
        diferencaEnergiaHorizonsXSimulacao.append(energiaHorizons[i]-energiaDoSistema[i])

        r1 = estado[0:2]
        v1 = estado[2:4]
        r2 = estado[4:6]
        v2 = estado[6:8]
        r3 = estado[8:10]
        v3 = estado[10:12]
        r_dinamicoSistema.append(np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)+np.sqrt((np.linalg.norm(r3-r1))**2 + epsilon**2)+np.sqrt((np.linalg.norm(r2-r3))**2 + epsilon**2))

        momLinSistema.append(calculaMomLin(m1simulacao,m2simulacao,m3simulacao,v1,v2,v3))
        momAngSistema.append(calculaMomAng(m1simulacao,m2simulacao,m3simulacao,r1,r2,r3,v1,v2,v3))

        #dados do horizons
        x1h,y1h,vx1h,vy1h,x2h,y2h,vx2h,vy2h,x3h,y3h,vx3h,vy3h=estadoHorizons

        r1h=np.array([x1h,y1h])
        r2h=np.array([x2h,y2h])
        r3h=np.array([x3h,y3h])
        r_dinamicoHorizons.append(np.sqrt((np.linalg.norm(r2h-r1h))**2 + epsilon**2)+np.sqrt((np.linalg.norm(r3h-r1h))**2 + epsilon**2)+np.sqrt((np.linalg.norm(r2h-r3h))**2 + epsilon**2))
        v1h=np.array([vx1h,vy1h])
        v2h=np.array([vx2h,vy2h])
        v3h=np.array([vx3h,vy3h])

        momLinHorizons.append(calculaMomLin(m1simulacao,m2simulacao,m3simulacao,v1h,v2h,v3h))
        momAngHorizons.append(calculaMomAng(m1simulacao,m2simulacao,m3simulacao,r1h,r2h,r3h,v1h,v2h,v3h))

        #append das diferenças
        diferencaMomLinHorizonsXSimulacao.append(momLinHorizons[i]-momLinSistema[i])
        diferencaMomAngHorizonsXSimulacao.append(momAngHorizons[i]-momAngSistema[i])

        diferencaDistanciaHorizonsXSimulacao.append(r_dinamicoHorizons[i]-r_dinamicoSistema[i])
        
        # diferença de posição relativa (das somas das posições relativas, na verdade --> inútil, porém interessante)
        r_rel_sim=(estado[4:6]-estado[0:2])+(estado[8:10]-estado[0:2])+(estado[8:10]-estado[4:6])  # simulador
        r_rel_hor=np.array([(xl[i]-xt[i])+(xl[i]-xs[i])+(xt[i]-xs[i]),(yl[i]-yt[i])+(yl[i]-ys[i])+(yt[i]-ys[i])])  #HORIZONS

        erro_posicao.append(np.linalg.norm(r_rel_sim-r_rel_hor))
        #erro_relativo_posicao=100*np.array(erro_posicao)/np.array(r_dinamicoSistema) #colocado fora do loop

        #evolução do sistema
        #estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao)
        for _ in range(N):
            estado=yoshida4ordem(estado,dt,m1simulacao,m2simulacao,m3simulacao)
        match flagTipoDeSimulacao:
            case 1: tAtual+=3600
            case 2: tAtual+=3600
            case 3: tAtual+=600
            case 4: tAtual+=3600
            case _: tAtual+=1

        #progresso da simulação em porcentagem 
        if passoAtual % 10000 == 0 and passoAtual > 0:
            print(f"Passo {passoAtual}/{steps} ({100*passoAtual/steps:.1f}%)")


    erro_relativo_posicao=100*np.array(erro_posicao)/np.array(r_dinamicoSistema)
    

    #==============VALIDANDO O SIMULADOR#==============
    massas=[m1simulacao,m2simulacao,m3simulacao]
    resultados=validarSimulador(trajetoria,massas,dt,calculaEnergiaDoSistema,calculaMomAng,calculaMomLin,yoshida4ordem)
    #resultados = [desvioEnerg,desvioLin,desvioAng,erroReversao]
    print(f"VALORES DA VALIDAÇÃO:\nDesvio da energia: {resultados[0]}\nDesvio do momento linear: {resultados[1]}\nDesvio do momento angular: {resultados[2]}\nErro da reversão: {resultados[3]:.2e}")

    
    #==========================================GERAÇÃO DE PLOTS EM UMA MESMA IMAGEM==========================================
    trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos
    


    
    #plotando a imagem da trajetória
    x1=trajetoria[:,0]
    y1=trajetoria[:,1]

    x2=trajetoria[:,4]
    y2=trajetoria[:,5]

    x3=trajetoria[:,8]
    y3=trajetoria[:,9]

    fig, axs = plt.subplots(4, 2, figsize=(12, 12))

    # ---------------- TRAJETÓRIA ----------------
    axs[0,0].plot(x1, y1, label='corpo 1')
    axs[0,0].plot(x2, y2, label='corpo 2')
    axs[0,0].plot(x3, y3, label='corpo 3')
    axs[0,0].scatter(x1[0], y1[0])
    axs[0,0].scatter(x2[0], y2[0])
    axs[0,0].scatter(x3[0], y3[0])
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
