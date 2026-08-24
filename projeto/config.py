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

#============================== META DE GERAÇÃO =================================
TOTALSIMULACOES_3_CORPOS=10
TOTALSIMULACOES_2_CORPOS=10

#============================== CAMINHOS =================================
PASTA_SAIDA_3_CORPOS="simulacoesArtificiais/simulacoes3C"
PASTA_SEEDS_3_CORPOS="seedsUsadas3C.txt"

PASTA_SAIDA_2_CORPOS="simulacoesArtificiais/simulacoes2C"
PASTA_SEEDS_2_CORPOS="seedsUsadas2C.txt"

#============================== CONDIÇÕES DE GERAÇÃO =================================
SALVAR_A_CADA=40        #salva 1 a cada 40 passos
MARGEM_SEGURANCA=25     #25 * 40 = 1000 passos antes de haver colisão é a margem