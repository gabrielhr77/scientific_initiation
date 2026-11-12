import numpy as np


#============================== VARIÁVEIS DA SIMULAÇÃO =================================
steps=400000
dt=0.00025
G=1
epsilon=1e-6

#============================== CONDIÇÃO INICIAL DA SIMULAÇÃO =================================
ESTADO0_3_CORPOS=np.array([
    -10, -5, 6, .11, .5, .4,
     10, -10, 3, -.2, -.6, .8,
     5, 10, -8, -.55, 0 , .3
],dtype=float)

ESTADO0_2_CORPOS=np.array([
    -10, -5, 6, .11, .5, .4,
     10, -10, 3, -.2, -.6, .8,
],dtype=float)

RAZAO_MASSA_MAX = 6.0

#============================== META DE GERAÇÃO =================================
TOTALSIMULACOES_3_CORPOS=4000
TOTALSIMULACOES_2_CORPOS=4000
TOTALSIMULACOES_3_CORPOS=300
TOTALSIMULACOES_2_CORPOS=100

#============================== CAMINHOS =================================
PASTA_SAIDA_3_CORPOS="simulacoesArtificiais/simulacoes3C"
PASTA_SEEDS_3_CORPOS="seedsUsadas3C.txt"

PASTA_SAIDA_2_CORPOS="simulacoesArtificiais/simulacoes2C"
PASTA_SEEDS_2_CORPOS="seedsUsadas2C.txt"

#============================== CONDIÇÕES DE GERAÇÃO =================================
SALVAR_A_CADA=40        #salva 1 a cada 40 passos
MARGEM_SEGURANCA=25     #25 * 40 = 1000 passos antes de haver colisão é a margem

#============================== 3 CORPOS NORMALIZADO =================================
#unidades: G=1, M=m1+m2+m3=1, E=-1/4 (tempo de travessia ~2.83)
E_ALVO_3C=-0.25
DT_3C=0.001              #passo do Yoshida
SALVAR_A_CADA_3C=50      #dt entre amostras = 0.05  -> no treino: AMOSTRAGEM=1 e DT_EFETIVO=0.05
STEPS_3C=100000          #duração = 100 unidades (~35 travessias)
R_ENCONTRO_3C=0.25            #encontro: para a simulação
R_FUGA_3C=30.0           #distância entre um par acima disso: descarta
MARGEM_3C=20             #amostras removidas antes do encontro
N_MIN_3C=300             #trajetória mais curta que isso é descartada
#PASTA_SAIDA_3_CORPOS="simulacoesArtificiais/simulacoes3N"
qMin3C=0.25            #faixa da razão virial K/|U| sorteada nas condições iniciais
qMax3C=0.9

#============================== 2 CORPOS NORMALIZADO =================================
E_ALVO_2C=-0.25
DT_2C=0.001              
SALVAR_A_CADA_2C=50      
STEPS_2C=100000          
R_ENCONTRO_2C=0.25            
R_FUGA_2C=30.0           
MARGEM_2C=20             
N_MIN_2C=300             
qMin2C=0.25            
qMax2C=0.9
E_ALVO_3C=-0.25
DT_3C=0.001
SALVAR_A_CADA_3C=50
STEPS_3C=100000
R_ENC_3C=0.25
R_FUGA_3C=30.0
MARGEM_3C=20
N_MIN_3C=300
Q_MIN_3C=0.25
Q_MAX_3C=0.9
