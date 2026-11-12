'''import torch as t

dados=t.load("redes/rede2/modelo3corpos.pt",map_location="cuda")
print(dados.keys())

a=dados["state_dict"]
b=dados["media"]
c=dados["desvio"]
d=dados["mediaMassa"]
e=dados["desvioMassa"]
#for name, tensor in dados.items():
#	print(f"{name:40s} {tuple(t.shape)}")
print("media:",b)
print("desvio:",c)
print("state_dict:",a)
print("mediaMassa:",d)
print("desvioMassa:",e)
'''
"""
testarModelo.py

Script para carregar um modelo já treinado (modelo3corpos.pt) e testá-lo
contra uma trajetória real (um arquivo .npz no mesmo formato usado no treino),
usando rollout autoregressivo.

Uso:
    python testarModelo.py caminho/para/trajetoria_real.npz

Requisitos:
    - Este arquivo deve estar na mesma pasta que 'saturn1.py' (ou o nome que
      você deu ao seu script de treino), pois importa funções/classes de lá.
    - 'saturn1.py' precisa ter o treino protegido por
      'if __name__ == "__main__":' (o seu já está assim), senão importar
      esse módulo dispararia o treino inteiro de novo.
"""

import sys
import numpy as np
import torch as t

# ---------------------------------------------------------------------
# AJUSTE AQUI: troque 'saturn1' pelo nome real do seu arquivo de treino
# (sem a extensão .py). Ex: se o arquivo é 'redeMark1.py', use:
#     from redeMark1 import (...)
# ---------------------------------------------------------------------

from versoes_treino_redes import saturn1
''' as saturn1
from saturn1 import (
    RedeTransformer,
    normalizar,
    desnormalizar,
    rolloutAutoregressivo,
    avaliarRollout,
)'''

# ================================================================
# HIPERPARÂMETROS DA ARQUITETURA
# Precisam ser EXATAMENTE os mesmos usados no treino, senao o
# load_state_dict falha (dimensões incompatíveis)
# ================================================================
D_MODEL = 64
N_HEADS = 4
N_CAMADAS = 6
DIM_FEEDFORWARD = 256
DROPOUT = 0.1
TAMANHO_JANELA = 10

# Quantos passos futuros prever no rollout autoregressivo
N_PASSOS_ROLLOUT = 300


def carregarModelo(caminhoCheckpoint, device):
    """Carrega o checkpoint completo (modelo3corpos.pt) e reconstrói o modelo."""
    pacote = t.load(caminhoCheckpoint, map_location=device)

    media = pacote["media"]
    desvio = pacote["desvio"]
    mediaMassa = pacote["mediaMassa"]
    desvioMassa = pacote["desvioMassa"]

    modelo = saturn1.RedeTransformer(
        dModel=D_MODEL,
        nHeads=N_HEADS,
        nCamadas=N_CAMADAS,
        dimFeedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
    ).to(device)

    modelo.load_state_dict(pacote["state_dict"])
    modelo.eval()  # desliga dropout e afins

    return modelo, media, desvio, mediaMassa, desvioMassa


def carregarTrajetoriaReal(caminhoNpz):
    """Carrega um único arquivo .npz real, no mesmo formato usado no treino."""
    dados = np.load(caminhoNpz)
    estado = dados["estado"]
    massas = dados["massas"]

    if estado.ndim < 2 or estado.shape[0] == 0:
        raise ValueError(f"Trajetória inválida ou vazia em: {caminhoNpz}")

    estado = estado[:, :18]  # garante só posições e velocidades
    return estado, massas


def main():
    if len(sys.argv) < 2:
        #print("Uso: python testarModelo.py caminho/para/trajetoria_real.npz")
        sys.exit(1)

    caminhoTrajetoria = sys.argv[1]
    caminhoCheckpoint = sys.argv[2] if len(sys.argv) >= 3 else "modelo3corpos.pt"   
    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"Rodando em: {device}")

    # ---------------- Carrega modelo + estatísticas de normalização ----------------
    modelo, media, desvio, mediaMassa, desvioMassa = carregarModelo(caminhoCheckpoint, device)
    print(f"Modelo carregado de '{caminhoCheckpoint}'")

    # ---------------- Carrega a trajetória real ----------------
    trajetoriaReal, massaReal = carregarTrajetoriaReal(caminhoTrajetoria)
    
    print(f"Trajetória real carregada: {trajetoriaReal.shape[0]} passos")

    nPassos = min(N_PASSOS_ROLLOUT, trajetoriaReal.shape[0] - TAMANHO_JANELA)
    if nPassos <= 0:
        raise ValueError(
            "Trajetória real curta demais: precisa de pelo menos "
            f"TAMANHO_JANELA + 1 = {TAMANHO_JANELA + 1} passos."
        )

    # ---------------- Normaliza com as MESMAS estatísticas do treino ----------------
    trajNorm = saturn1.normalizar(trajetoriaReal, media, desvio)
    massaNorm = saturn1.normalizar(massaReal, mediaMassa, desvioMassa)

    # ---------------- Rollout autoregressivo ----------------
    previstoNorm = saturn1.rolloutAutoregressivo(
        modelo, trajNorm, massaNorm, TAMANHO_JANELA, nPassos, device
    )
    previstoReal = saturn1.desnormalizar(previstoNorm, media, desvio).numpy()

    alvoReal = trajetoriaReal[TAMANHO_JANELA: TAMANHO_JANELA + nPassos]

    # ---------------- Métrica simples de erro ----------------
    erro = np.linalg.norm(previstoReal - alvoReal, axis=1)
    print(f"\nErro médio (rollout de {nPassos} passos): {erro.mean():.6f}")
    print(f"Erro no primeiro passo: {erro[0]:.6f}")
    print(f"Erro no último passo:   {erro[-1]:.6f}")

    # ---------------- Gráfico + horizonte de previsibilidade ----------------
    # Reaproveita a função avaliarRollout já pronta, passando só esta trajetória.
    saturn1.avaliarRollout(
        modelo,
        [trajetoriaReal],
        [massaReal],
        TAMANHO_JANELA,
        media,
        desvio,
        mediaMassa,
        desvioMassa,
        device,
        nTrajetoriasMax=1,
        nPassosDoRollout=nPassos,
    )


if __name__ == "__main__":
    main()