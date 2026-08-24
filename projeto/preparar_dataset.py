import os, numpy as np
from config import *

def salvarEstadosNPZ(caminho,massas,estado,tempo,energia,momAng,momLin,nome,dt,motivoTermino):
    #convertendo para array caso ainda não seja
    massas=np.array(massas)
    estado=estado
    tempo=np.array(tempo)
    energia=np.array(energia)
    momAng=np.array(momAng)
    momLin=np.array(momLin)
    motivoTermino=motivoTermino
    
    np.savez_compressed(caminho+f"simulacao3D_{nome}.npz",
                        massas=massas,estado=estado,tempo=tempo,energia=energia,momAng=momAng,momLin=momLin,dt=dt,motivoTermino=motivoTermino)
    print(f"Arquivo da seed {nome} salvo. Tamanho aproximado: {estado.nbytes/1024:.1f} KB")

def carregarSeedsUsadas(caminho):
    if not os.path.exists(caminho):
        return set() #o que seria esse SET()?
    with open(caminho) as file:
        return {int(linha.strip()) for linha in file if linha.strip()} #retorna um vetor com as linhas salvas

def proximoBlocoSeeds(qtdd,caminho):
    usadas=carregarSeedsUsadas(caminho)
    proximo=(max(usadas)+1) if usadas else 0
    return list(range(proximo, proximo+qtdd))

#retorna um valor aleatório de um parâmetro (multiplica por algum valor entre 0 e 1), podendo ser negativo ou positivo
def rd(a):
    return float(a)*(2*np.random.rand()-1)

def carregarSimulacao():
    return

