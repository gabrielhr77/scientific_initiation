#============================== 3 CORPOS =================================

def converter_V_para_P_3Corpos(m1,m2,m3,estado): #usado após ser gerado pelo yoshida, no loop
    estadoAux=estado.copy()
    estadoAux[3:6] *= m1
    estadoAux[9:12] *= m2
    estadoAux[15:18] *= m3
    return estadoAux

def converter_P_para_V_3Corpos(m1,m2,m3,estado): #usado após ser salvo vetor de estados para retornar para velocidades para manter o cálculo da LOSS de maneira correta
    estadoAux=estado.copy()
    estadoAux[3:6] /= m1
    estadoAux[9:12] /= m2
    estadoAux[15:18] /= m3
    return estadoAux


#============================== 2 CORPOS =================================
def converter_V_para_P_2Corpos(m1,m2,estado): #usado após ser gerado pelo yoshida, no loop
    estadoAux=estado.copy()
    estadoAux[3:6] *= m1
    estadoAux[9:12] *= m2
    return estadoAux

def converter_P_para_V_2Corpos(m1,m2,estado): #usado após ser salvo vetor de estados para retornar para velocidades para manter o cálculo da LOSS de maneira correta
    estadoAux=estado.copy()
    estadoAux[3:6] /= m1
    estadoAux[9:12] /= m2
    return estadoAux