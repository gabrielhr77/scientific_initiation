import numpy as np, matplotlib.pyplot as plt

'''
OBS1: o objetivo deste código é gerar a simulação (posições e velocidades) do sistema triplo KOI-126 apenas com os dados orbitais presentes na internet sobre o sistema

Antes de cada função há uma explicação sobre seu funcionamento

PARÂMETROS DAS ÓRBITAS KEPLERIANAS
a   semieixo maior (tamanho da órbita)
e   excentricidade (e=0 => círculo)
i   inclinação do plano orbital
ω   argumento do periastro (orientação da elipse no plano)
M   anomalia média (posição do corpo na época)

OBS2: devemos lembrar que KOI-126 é um sistema de 3 corpos hierárquico com uma estrela A maior (orbita o centro de massa do par BC),
com o par BC (B orbitando C ou vice-versa e com seu centro de massa orbitando A)
'''


def inicializar_koi126(m1, m2, m3, G):
    """
    Retorna estado inicial [x1,y1,vx1,vy1, x2,y2,vx2,vy2, x3,y3,vx3,vy3]
    para o sistema KOI-126 na época T0

    Converto para radianos pois as funções trigonométricas em python usam radianos
    
    Mantenho o centro de massa do sistema na origem
    """
    #Parâmetros da binária interna (B+C)
    a_binaria = 0.021986  #semieixo maior em UA - muito perto
    e_binaria = 0.02234   #sem unidade - é quase circular
    ω_binaria = np.radians(8.012)   #dados do artigo
    M_binaria = np.radians(355.66)    #dados do artigo
    
    #Parâmetros da órbita externa (A em torno do baricentro BC)
    a_sistema = 0.2495  #tambem em UA , bem mais distante
    e_sistema = 0.3043  #BEM excêntrica
    ω_sistema = np.radians(52.88)   #dados do artigo
    M_sistema = np.radians(19.87)   #dados do artigo
    
    #Resolver órbita interna, onde r_rel é a posição de C em relação a B ou vice-versa, retornando o vetor do corpo 2 em relação ao corpo 1
    r_rel, v_rel = resolver_orbita_binaria(
        a_binaria, e_binaria, ω_binaria, M_binaria, m2, m3, G
    )
    
    #Massa do par BC
    M_BC = m2 + m3
    
    #Posições no referencial do baricentro BC, o que é importante pois, num sistema binário, ambos os corpos orbitam o CENTRO DE MASSA, então calculo a posição e velocidade relativa de cada corpo com relação ao centro de massa
    r2_BC = (m3 / M_BC) * r_rel
    r3_BC = (-m2 / M_BC) * r_rel
    v2_BC = (m3 / M_BC) * v_rel
    v3_BC = (-m2 / M_BC) * v_rel
    
    #Resolver órbita externa (A em relação ao baricentro BC)
    r_A_BC, v_A_BC = resolver_orbita_binaria(
        a_sistema, e_sistema, ω_sistema, M_sistema, M_BC, m1, G
    )
    
    #Massa total
    M_total = m1 + m2 + m3
    
    #Posição do baricentro BC em relação ao CM total - este é o mesmo raciocínio do cálcuko de r2_BC e r3_BC, mas agora o corpo 1 é A e o corpo 2 é o centro de massa de BC
    r_BC_CM = (-m1 / M_total) * r_A_BC
    v_BC_CM = (-m1 / M_total) * v_A_BC
    
    #Posição de A em relação ao CM total
    r1 = r_A_BC + r_BC_CM  #(A em relação a BC)+(BC em relação ao CM)
    v1 = v_A_BC + v_BC_CM
    
    #Posições de B e C
    r2 = r_BC_CM + r2_BC
    r3 = r_BC_CM + r3_BC
    v2 = v_BC_CM + v2_BC
    v3 = v_BC_CM + v3_BC
    
    #Montar vetor de estado
    estado = np.array([
        r1[0], r1[1], v1[0], v1[1],
        r2[0], r2[1], v2[0], v2[1],
        r3[0], r3[1], v3[0], v3[1]
    ], dtype=float)
    
    return estado

'''
EXPLICAÇÃO:
* a função resolve uma órbita kepleriana, retornando posição e velocidade relativas entre os dois corpos em questão
* M é a anomalia média, que é uma medida angular que representa a fração do período orbital que passou desde a passagem pelo PERIASTRO -ponto mais próximo. Aumenta linearmente com o tempo se a órbita for circular e de velocidade cte. Diretamente proporcional à área varrida e corresponde ao ÂNGULO QUE O OBJETO TERIA PERCORRIDO CASO SUA ÓRBITA FOSSE CIRCULAR (vel angular cte) e não elíptica
* E é a anomalia excentrica, usada para calcular posições reais --> M = E -e*sin(E), usando newton-rapson par achar E numericamente
* é como se M fosse a posição do ponteiro de um relógio, mas E é a posição real do ponteiro, que se move mais rápido no periélio

* x_orb = a*(np.cos(E) - e) e y_orb = a*np.sqrt(1 - e**2)*np.sin(E) são as equações paramétricas da elipse, com a 1º sendo a distância do foco da elipse (onde está a estrela) -que está em (-ae,0)- ao corpo e a 2º a coordenada perpendicular
'''
def resolver_orbita_binaria(a, e, ω, M, m1, m2, G):
    """
    Versão simplificada para 2D (ignora INCLINAÇÃO ORBITAL e Ω)
    """
    #Resolver anomalia excêntrica (E)
    E = M
    for _ in range(10):
        f = E - e*np.sin(E) - M
        f_prime = 1 - e*np.cos(E)
        E -= f/f_prime
    
    #Posição na órbita (coordenadas da órbita) 
    x_orb = a*(np.cos(E) - e)
    y_orb = a*np.sqrt(1 - e**2)*np.sin(E)
    
    #Rotacionar pelo argumento do periastro (ω) - a rotação por este argumento faz com que o PERIASTRO (ponto de maior aproximação) mude de lugar ao longo do tempo
    cosω = np.cos(ω)
    sinω = np.sin(ω)
    x = x_orb*cosω - y_orb*sinω
    y = x_orb*sinω + y_orb*cosω
    
    #Velocidade na órbita
    n = np.sqrt(G*(m1 + m2)/a**3) #MOVIMENTO MÉDIO, que é a velocidade angular n=2π/p, t.q. derivando da 3ª lei de kepler p²=4π²a³/(G(m)) temos que n=(G(M)/a³)**0.5    
    cosE = np.cos(E)
    sinE = np.sin(E)
    factor = a*n / (1 - e*cosE) #FATOR DE ESCALA: converte a posição em velocidade
    vx_orb = -factor * sinE
    vy_orb = factor * np.sqrt(1 - e**2) * cosE
    
    #Rotacionar velocidade - aqui é para termos a velocidade no plano
    vx = vx_orb*cosω - vy_orb*sinω
    vy = vx_orb*sinω + vy_orb*cosω
    
    return np.array([x, y]), np.array([vx, vy])


def salvarEstadosNPZ(trajetoria,energia,momAng,momLin,r_min,nome):
    #convertendo para array caso ainda não seja
    trajetoria=np.array(trajetoria)
    energia=np.array(energia)
    momAng=np.array(momAng)
    momLin=np.array(momLin)
    r_min=np.array(r_min)

    #para facilitar no treinamento da rede, salvo todos os arrays em um arquivo compactado
    np.savez_compressed(f"simulacoesArtificiais/simulacao{nome}.npz",
                        trajetoria=trajetoria,energia=energia,momAng=momAng,momLin=momLin,r_min=r_min)
    print(f"Arquivo salvo. Tamanho aproximado: {trajetoria.nbytes/1e6:.1f} MB")


#função para carregar os dados depois
def carregarEstadosNPZ(nome):
    dados=np.load(f"simulacoesArtificiais/simulacao{nome}.npz")
    return dados['trajetoria']




#VERSÃO APRIMORADA, SEM O USO DO CONCATENATE, QUE GERA UM VETOR POR PASSO DO VERLET
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


def calculaEnergiaEMomentoDoSistema(estado,m1,m2,m3): #e momentos linear e angular
    x1,y1,vx1,vy1,\
    x2,y2,vx2,vy2,\
    x3,y3,vx3,vy3=estado

    r1=np.array([x1,y1])
    r2=np.array([x2,y2])
    r3=np.array([x3,y3])

    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)
    r13=np.sqrt((np.linalg.norm(r3-r1))**2 + epsilon**2)
    r23=np.sqrt((np.linalg.norm(r3-r2))**2 + epsilon**2)

    v1=np.array([vx1,vy1])
    v2=np.array([vx2,vy2])
    v3=np.array([vx3,vy3])

    massas=np.array([m1,m2,m3])

    momLin.append(np.linalg.norm(m1*v1 + m2*v2 + m3*v3))

    momAng.append(m1*(r1[0]*v1[1] - r1[1]*v1[0]) + m2*(r2[0]*v2[1] - r2[1]*v2[0]) + m3*(r3[0]*v3[1] - r3[1]*v3[0]))

    cinetica=((vx1**2+vy1**2)*m1/2+(vx2**2+vy2**2)*m2/2+(vx3**2+vy3**2)*m3/2)
    
    potencial=-G*(m1*m2/r12+m1*m3/r13+m2*m3/r23)
    
    r_min.append(min(r12,r13,r23))

    return cinetica+potencial





#loop temporal para a simulação
'''estado0=np.array([
    -10, -5, .11, 0.05,
     1, -10, -.2, -0.06,
     5, 10, -0.055, 0
],dtype=float)
'''



#============================== VARIÁVEIS DA SIMULAÇÃO =================================
passoAtual=0
epsilon=1e-12 #para ser uma simulação fisicamente acurada/real essa variável tem que ser praticamente ZERO
dt=0.0005
steps = int(10 * 33.9214 / dt)
m1 = 1.2713
m2 = 0.23529
m3 = 0.20739
G=2.959122082855911e-4


estado0 = inicializar_koi126(m1, m2, m3, G)

# Ajustar para centro de massa zero
x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = estado0

r1 = np.array([x1, y1])
r2 = np.array([x2, y2])
r3 = np.array([x3, y3])
v1 = np.array([vx1, vy1])
v2 = np.array([vx2, vy2])
v3 = np.array([vx3, vy3])

# Calcular CM
M_total = m1 + m2 + m3
R_cm = (m1*r1 + m2*r2 + m3*r3) / M_total
V_cm = (m1*v1 + m2*v2 + m3*v3) / M_total

# Ajustar posições e velocidades
r1 -= R_cm
r2 -= R_cm
r3 -= R_cm
v1 -= V_cm
v2 -= V_cm
v3 -= V_cm

# CRIAR NOVO ESTADO COM VALORES AJUSTADOS!
estado = np.array([
    r1[0], r1[1], v1[0], v1[1],
    r2[0], r2[1], v2[0], v2[1],
    r3[0], r3[1], v3[0], v3[1]
], dtype=float)

# Listas para armazenar dados
trajetoria = []
energiaDoSistema = []
r_min = []
momLin = []
momAng = []

print("Iniciando simulação do KOI-126...")
print(f"dt = {dt} dias (~{dt*24*60:.2f} minutos)")
print(f"Steps = {steps}")
print(f"Tempo total = {steps*dt:.1f} dias = {steps*dt/365.25:.2f} anos")

# Loop principal
for i in range(steps):
    passoAtual += 1
    trajetoria.append(estado.copy())
    energiaDoSistema.append(calculaEnergiaEMomentoDoSistema(estado, m1, m2, m3))
    
    # Avançar no tempo
    estado = yoshida4ordem(estado, dt, m1, m2, m3)
    
    # Verificar colisões (usando raios estelares reais)
    r12 = np.linalg.norm(estado[4:6] - estado[0:2])
    r13 = np.linalg.norm(estado[8:10] - estado[0:2])
    r23 = np.linalg.norm(estado[8:10] - estado[4:6])
    
    # Raios em UA (convertidos de R☉)
    R_sol_ua = 4.65047e-3
    R1 = 1.9984 * R_sol_ua
    R2 = 0.25504 * R_sol_ua
    R3 = 0.23196 * R_sol_ua
    
    # Distância mínima segura (soma dos raios)
    if r12 < (R1 + R2) or r13 < (R1 + R3) or r23 < (R2 + R3):
        print(f"COLISÃO DETECTADA no passo {passoAtual}")
        print(f"Distâncias: r12={r12:.6f}, r13={r13:.6f}, r23={r23:.6f} UA")
        print(f"Raios: R1+R2={R1+R2:.6f}, R1+R3={R1+R3:.6f}, R2+R3={R2+R3:.6f} UA")
        break
    
    # Progresso
    if i % 100000 == 0 and i > 0:
        print(f"Passo {i}/{steps} ({100*i/steps:.1f}%)")

# Converter para array
trajetoria = np.array(trajetoria)
print(f"\nSimulação completa! {len(trajetoria)} passos")
    

#salvando em NPZ
salvarEstadosNPZ(trajetoria,energiaDoSistema,momAng,momLin,r_min,"Koi126")

trajetoria=np.array(trajetoria)#aqui o trajetoria deixa de ser uma lista de arrays para ser uma matriz 2D, melhor para fazer cálculos

print(trajetoria.shape)

#plotando a imagem da trajetória
x1=trajetoria[:,0]
y1=trajetoria[:,1]

x2=trajetoria[:,4]
y2=trajetoria[:,5]

x3=trajetoria[:,8]
y3=trajetoria[:,9]


#GERAÇÃO DE PLOTS EM UMA MESMA IMAGEM

fig, axs = plt.subplots(3, 2, figsize=(12, 12))

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

# ---------------- r_min ----------------
axs[0,1].plot(r_min)
axs[0,1].set_title("Distância mínima em UA")
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