#!/usr/bin/env python3
"""
Transformer para 3 corpos — hipótese A: base Yoshida(n=1, dt=DT_EFETIVO) + resíduo.

A rede não aprende o passo inteiro. A cabeça (zerada no início) prevê só a correção
sobre um único passo Yoshida de tamanho 0.05. O alvo da loss continua sendo
integrar(..., n=50) a partir do estado de entrada.
"""
import numpy as np
import torch as t
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
import re
from tqdm import tqdm
from config import *

G = 1.0
epsilon = 1e-6

# hipótese A
FRACAO_ESCALA_RESID = 0.02
N_YOSHIDA_BASE = 1
PREFIXO_CKPT = "checkpointResidEpoca"


# ================================================================ DADOS ================================================================
def carregarTrajetorias(pastaNPZ):
    trajetorias, massasLista = [], []
    for arquivo in sorted(Path(pastaNPZ).glob("*.npz")):
        dados = np.load(arquivo)
        estado = dados["estado"]
        if estado.ndim < 2 or estado.shape[0] == 0:
            continue
        estado = estado[:, :18]
        trajetorias.append(estado)
        massasLista.append(dados["massas"])
    return trajetorias, massasLista

def acharUltimoCheckpoint(pasta=".", prefixo=PREFIXO_CKPT):
    checkpoints = sorted(Path(pasta).glob(f"{prefixo}*.pt"))
    if not checkpoints:
        return None, 0
    ultimo = checkpoints[-1]
    epoca = int(re.search(r"(\d+)", ultimo.stem).group(1))
    return ultimo, epoca + 1

def calcularEscalaDelta(trajetorias, amostragem, media, desvio):
    ds = []
    for traj in trajetorias:
        x = normalizar(traj[::amostragem], media, desvio)
        ds.append(x[1:] - x[:-1])
    return t.cat(ds).std(dim=0).clamp_min(1e-8)

def calcularEscalasFisicas(trajetorias, massas, amostragem):
    escalaEnergia, momIndividuais, escalaMomAng = [], [], []
    for traj, m in zip(trajetorias, massas):
        a = t.as_tensor(traj[::amostragem], dtype=t.float32)
        mt = t.as_tensor(m, dtype=t.float32).unsqueeze(0).expand(len(a), 3)
        escalaEnergia.append(calcularHamiltoniano(a, mt))
        v1, v2, v3 = a[..., 3:6], a[..., 9:12], a[..., 15:18]
        m1, m2, m3 = mt[..., 0:1], mt[..., 1:2], mt[..., 2:3]
        momIndividuais.append(t.cat([m1 * v1, m2 * v2, m3 * v3], dim=0))
        escalaMomAng.append(calcularMomentoAngular(a, mt))
    escalaEnergia = t.cat(escalaEnergia).abs().mean().clamp_min(1e-8)
    escalaMomLin = t.cat(momIndividuais).std().clamp_min(1e-8)
    escalaMomAng = t.cat(escalaMomAng).reshape(-1, 3).std().clamp_min(1e-8)
    return escalaEnergia, escalaMomLin, escalaMomAng

def calcularNormalizacao(trajetorias):
    todos = np.concatenate(trajetorias)
    media = todos.mean(axis=0)
    desvio = todos.std(axis=0)
    desvio[desvio < 1e-8] = 1.0
    return media, desvio

def normalizar(x, media, desvio):
    x = t.as_tensor(x, dtype=t.float32)
    media = t.as_tensor(media, dtype=t.float32, device=x.device)
    desvio = t.as_tensor(desvio, dtype=t.float32, device=x.device)
    return (x - media) / desvio

def desnormalizar(x, media, desvio):
    x = t.as_tensor(x, dtype=t.float32)
    media = t.as_tensor(media, dtype=t.float32, device=x.device)
    desvio = t.as_tensor(desvio, dtype=t.float32, device=x.device)
    return x * desvio + media

def horizonteAtual(epoca, horizonteMax, nEpocas):
    if epoca / nEpocas < 0.15:
        return 1
    else:
        horizonteMin = int(horizonteMax / 10)
        epocasParaMaximo = nEpocas * 0.8
        fracao = min(1.0, epoca / epocasParaMaximo)
        return int(horizonteMin + fracao * (horizonteMax - horizonteMin))

class JanelaComHorizonte(Dataset):
    def __init__(self, trajetorias, massas, tamanhoJanela, horizonteMax, amostragem, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.horizonteMax = horizonteMax
        self.amostragem = amostragem
        self.trajetorias = [normalizar(traj[::amostragem], media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]
        comprimentoTotal = tamanhoJanela + horizonteMax
        self.indices = []
        for idTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - comprimentoTotal + 1
            pulo = 5
            for i in range(0, max(nAmostras, 0), pulo):
                self.indices.append((idTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idTraj, i = self.indices[idx]
        traj = self.trajetorias[idTraj]
        janela = traj[i:i + self.tamanhoJanela]
        futuros = traj[i + self.tamanhoJanela:i + self.tamanhoJanela + self.horizonteMax]
        massa = self.massas[idTraj]
        return (janela.clone(), massa.clone(), futuros.clone())

class JanelaProximoEstado(Dataset):
    def __init__(self, trajetorias, massas, tamanhoJanela, amostragem, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.amostragem = amostragem
        self.trajetorias = [normalizar(traj[::amostragem], media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]
        self.indices = []
        for idTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - tamanhoJanela
            pulo = 5
            for i in range(0, max(nAmostras, 0), pulo):
                self.indices.append((idTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idTraj, i = self.indices[idx]
        traj = self.trajetorias[idTraj]
        janela = traj[i:i + self.tamanhoJanela]
        alvo = traj[i + self.tamanhoJanela]
        massa = self.massas[idTraj]
        return (janela.clone(), massa.clone(), alvo.clone())


# ================================================================ FÍSICA ================================================================
def calcularHamiltoniano(estado, massas):
    r1, v1 = estado[..., 0:3], estado[..., 3:6]
    r2, v2 = estado[..., 6:9], estado[..., 9:12]
    r3, v3 = estado[..., 12:15], estado[..., 15:18]
    m1, m2, m3 = massas[..., 0], massas[..., 1], massas[..., 2]
    r12 = t.sqrt(t.linalg.norm(r2 - r1, dim=-1) ** 2 + epsilon ** 2)
    r13 = t.sqrt(t.linalg.norm(r3 - r1, dim=-1) ** 2 + epsilon ** 2)
    r23 = t.sqrt(t.linalg.norm(r3 - r2, dim=-1) ** 2 + epsilon ** 2)
    energiaCineticaSistema = (
        0.5 * m1 * t.sum(v1 ** 2, dim=-1)
        + 0.5 * m2 * t.sum(v2 ** 2, dim=-1)
        + 0.5 * m3 * t.sum(v3 ** 2, dim=-1)
    )
    energiaPotencialSistema = -G * (m1 * m2 / r12 + m1 * m3 / r13 + m2 * m3 / r23)
    return energiaCineticaSistema + energiaPotencialSistema

def calcularMomentoLinearVetor(estado, massas):
    v1, v2, v3 = estado[..., 3:6], estado[..., 9:12], estado[..., 15:18]
    m1, m2, m3 = massas[..., 0], massas[..., 1], massas[..., 2]
    return (m1[..., None] * v1 + m2[..., None] * v2 + m3[..., None] * v3)

def calcularMomentoLinear(estado, massas):
    return t.linalg.norm(calcularMomentoLinearVetor(estado, massas), dim=-1)

def calcularMomentoAngular(estado, massas):
    r1, v1 = estado[..., 0:3], estado[..., 3:6]
    r2, v2 = estado[..., 6:9], estado[..., 9:12]
    r3, v3 = estado[..., 12:15], estado[..., 15:18]
    m1, m2, m3 = massas[..., 0], massas[..., 1], massas[..., 2]
    return (
        m1[..., None] * t.cross(r1, v1, dim=-1)
        + m2[..., None] * t.cross(r2, v2, dim=-1)
        + m3[..., None] * t.cross(r3, v3, dim=-1)
    )

def atualizaAceleracoes_posicoes3Corpos(r1, r2, r3, m1, m2, m3):
    pr12, pr13, pr23 = r2 - r1, r3 - r1, r3 - r2
    rr12 = t.sum(pr12 ** 2, dim=-1) + epsilon ** 2
    rr13 = t.sum(pr13 ** 2, dim=-1) + epsilon ** 2
    rr23 = t.sum(pr23 ** 2, dim=-1) + epsilon ** 2
    inv_r12 = 1 / (rr12 * t.sqrt(rr12))
    inv_r13 = 1 / (rr13 * t.sqrt(rr13))
    inv_r23 = 1 / (rr23 * t.sqrt(rr23))
    a1 = G * (m2[..., None] * pr12 * inv_r12[..., None] + m3[..., None] * pr13 * inv_r13[..., None])
    a2 = G * (-m1[..., None] * pr12 * inv_r12[..., None] + m3[..., None] * pr23 * inv_r23[..., None])
    a3 = G * (-m1[..., None] * pr13 * inv_r13[..., None] - m2[..., None] * pr23 * inv_r23[..., None])
    return a1, a2, a3

def integradorYoshida4aOrdem(estado, dt, m1, m2, m3):
    w1 = 1 / (2 - 2 ** (1 / 3))
    w0 = -2 ** (1 / 3) / (2 - 2 ** (1 / 3))
    c1 = c4 = w1 / 2
    c2 = c3 = (w0 + w1) / 2
    d1 = d3 = w1
    d2 = w0
    r1, v1 = estado[..., 0:3], estado[..., 3:6]
    r2, v2 = estado[..., 6:9], estado[..., 9:12]
    r3, v3 = estado[..., 12:15], estado[..., 15:18]
    r1 = r1 + c1 * dt * v1
    r2 = r2 + c1 * dt * v2
    r3 = r3 + c1 * dt * v3
    a1, a2, a3 = atualizaAceleracoes_posicoes3Corpos(r1, r2, r3, m1, m2, m3)
    v1 = v1 + d1 * dt * a1
    v2 = v2 + d1 * dt * a2
    v3 = v3 + d1 * dt * a3
    r1 = r1 + c2 * dt * v1
    r2 = r2 + c2 * dt * v2
    r3 = r3 + c2 * dt * v3
    a1, a2, a3 = atualizaAceleracoes_posicoes3Corpos(r1, r2, r3, m1, m2, m3)
    v1 = v1 + d2 * dt * a1
    v2 = v2 + d2 * dt * a2
    v3 = v3 + d2 * dt * a3
    r1 = r1 + c3 * dt * v1
    r2 = r2 + c3 * dt * v2
    r3 = r3 + c3 * dt * v3
    a1, a2, a3 = atualizaAceleracoes_posicoes3Corpos(r1, r2, r3, m1, m2, m3)
    v1 = v1 + d3 * dt * a1
    v2 = v2 + d3 * dt * a2
    v3 = v3 + d3 * dt * a3
    r1 = r1 + c4 * dt * v1
    r2 = r2 + c4 * dt * v2
    r3 = r3 + c4 * dt * v3
    return t.cat([r1, v1, r2, v2, r3, v3], dim=-1)

def _integrarEager(estado, dt, m1, m2, m3, n):
    for _ in range(n):
        estado = integradorYoshida4aOrdem(estado, dt / n, m1, m2, m3)
    return estado

class _IntegradorCUDAGraph:
    def __init__(self):
        self.cache = {}

    @t.no_grad()
    def __call__(self, estado, dt, m1, m2, m3, n):
        if estado.device.type != "cuda":
            return _integrarEager(estado, dt, m1, m2, m3, n)
        chave = (tuple(estado.shape), float(dt), n)
        if chave not in self.cache:
            sE = estado.clone()
            sm1, sm2, sm3 = m1.clone(), m2.clone(), m3.clone()
            s = t.cuda.Stream()
            s.wait_stream(t.cuda.current_stream())
            with t.cuda.stream(s):
                for _ in range(3):
                    _integrarEager(sE, dt, sm1, sm2, sm3, n)
            t.cuda.current_stream().wait_stream(s)
            g = t.cuda.CUDAGraph()
            with t.cuda.graph(g):
                sSaida = _integrarEager(sE, dt, sm1, sm2, sm3, n)
            self.cache[chave] = (g, sE, sm1, sm2, sm3, sSaida)
        g, sE, sm1, sm2, sm3, sSaida = self.cache[chave]
        sE.copy_(estado); sm1.copy_(m1); sm2.copy_(m2); sm3.copy_(m3)
        g.replay()
        return sSaida.clone()

_integradorGrafo = _IntegradorCUDAGraph()

def integrar(estado, dt, m1, m2, m3, n):
    return _integradorGrafo(estado, dt, m1, m2, m3, n)


# ================================================================ ARQUITETURA DA REDE ================================================================
class EmbeddingFisico(t.nn.Module):
    def __init__(self, dimEstado=18, dModel=64, bias=True):
        super().__init__()
        self.proj = t.nn.Linear(dimEstado, dModel, bias=bias)

    def forward(self, janela):
        return self.proj(janela)

class CodificacaoPosicional(t.nn.Module):
    def __init__(self, dModel, tamanhoJanelaMax=50):
        super().__init__()
        self.embeddPosicao = t.nn.Embedding(tamanhoJanelaMax, dModel)

    def forward(self, i):
        B, K, _ = i.shape
        posicoes = t.arange(K, device=i.device)
        posEmb = self.embeddPosicao(posicoes)
        return i + posEmb.unsqueeze(0)

def _buf(valor, nome):
    if valor is not None:
        return t.as_tensor(valor, dtype=t.float32)
    g = globals().get(nome)
    if g is None:
        raise RuntimeError(f"RedeTransformer precisa de {nome} (passe no construtor ou defina o global antes)")
    return t.as_tensor(g, dtype=t.float32)

class RedeTransformer(t.nn.Module):
    def __init__(self, dimEstado=18, dimMassa=3, dModel=64, nHeads=4, nCamadas=6, dimFeedforward=256,
                 dropout=0.0, tamanhoMaximoJanela=50, escalaDelta=None, escalaResid=None,
                 media=None, desvio=None, mediaMassa=None, desvioMassa=None, dtEfetivo=None,
                 nYoshidaBase=N_YOSHIDA_BASE):
        super().__init__()
        self.dimEstado = dimEstado
        self.dimMassa = dimMassa
        self.nYoshidaBase = nYoshidaBase
        self.register_buffer("escalaDelta", _buf(escalaDelta, "escalaDelta"))
        self.register_buffer("escalaResid", _buf(escalaResid, "escalaResid"))
        self.register_buffer("media", _buf(media, "mediaT"))
        self.register_buffer("desvio", _buf(desvio, "desvioT"))
        self.register_buffer("mediaMassa", _buf(mediaMassa, "mediaMassaT"))
        self.register_buffer("desvioMassa", _buf(desvioMassa, "desvioMassaT"))
        if dtEfetivo is None:
            dtEfetivo = globals().get("DT_EFETIVO", DT_3C * SALVAR_A_CADA_3C)
        self.register_buffer("dtEfetivo", t.tensor(float(dtEfetivo)))
        self.embedding = EmbeddingFisico(dimEstado + dimMassa, dModel)
        self.codPosicional = CodificacaoPosicional(dModel, tamanhoMaximoJanela)
        self.dropoutEntrada = t.nn.Dropout(dropout)
        camada = t.nn.TransformerEncoderLayer(
            d_model=dModel, nhead=nHeads, dim_feedforward=dimFeedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = t.nn.TransformerEncoder(camada, num_layers=nCamadas)
        self.cabecaSaida = t.nn.Linear(dModel, dimEstado)
        t.nn.init.zeros_(self.cabecaSaida.weight)
        t.nn.init.zeros_(self.cabecaSaida.bias)

    def forward(self, janela, massa):
        B, K, _ = janela.shape
        massaExpandida = massa.unsqueeze(1).expand(B, K, self.dimMassa)
        entradaRede = t.cat([janela, massaExpandida], dim=-1)
        i = self.dropoutEntrada(self.codPosicional(self.embedding(entradaRede)))
        saida = self.encoder(i)
        z = self.cabecaSaida(saida[:, -1, :])
        with t.no_grad():
            ultimoFis = (janela[:, -1, :] * self.desvio + self.media).contiguous()
            massaFis = (massa * self.desvioMassa + self.mediaMassa).contiguous()
            dt = float(self.dtEfetivo)
            baseFis = integrar(
                ultimoFis, dt,
                massaFis[:, 0].contiguous(),
                massaFis[:, 1].contiguous(),
                massaFis[:, 2].contiguous(),
                self.nYoshidaBase,
            )
        baseNorm = (baseFis - self.media) / self.desvio
        return baseNorm + z * self.escalaResid


# ================================================================ LOSS ================================================================
def lossEstado(previsto, real, escala):
    return t.mean(((previsto - real) / escala) ** 2)

def lossEnergia(previsto, real, massas, escalaEnergia):
    return ((calcularHamiltoniano(real, massas) - calcularHamiltoniano(previsto, massas)) / escalaEnergia) ** 2

def lossMomentoLinear(previsto, real, massas, escalaMomLin):
    diferenca = calcularMomentoLinearVetor(previsto, massas) - calcularMomentoLinearVetor(real, massas)
    return t.sum((diferenca / escalaMomLin) ** 2, dim=-1)

def lossMomentoAngular(previsto, real, massas, escalaMomAng):
    return (t.linalg.norm((calcularMomentoAngular(previsto, massas) - calcularMomentoAngular(real, massas)) / escalaMomAng, dim=-1)) ** 2

def lossConsistenciaYoshida(previsto, ultimoEstadoJanela, massas, dt, subpassos):
    with t.no_grad():
        a = ultimoEstadoJanela
        for _ in range(subpassos):
            a = integradorYoshida4aOrdem(a, dt / subpassos, massas[..., 0], massas[..., 1], massas[..., 2])
    return t.sum((previsto - a) ** 2, dim=-1)

def lossTotal(previsto, real, ultimoEstadoJanela, massas, dt, pesos, subpassos, escalaEstado, escalaEnergia, escalaMomLin, escalaMomAng):
    termosDaLOSS = {
        "estado": lossEstado(previsto, real, escalaEstado),
        "energia": lossEnergia(previsto, real, massas, escalaEnergia).mean(),
        "momentoLinear": lossMomentoLinear(previsto, real, massas, escalaMomLin).mean(),
        "momentoAngular": lossMomentoAngular(previsto, real, massas, escalaMomAng).mean(),
        "consistencia": (
            lossConsistenciaYoshida(previsto, ultimoEstadoJanela, massas, dt, subpassos).mean()
            if pesos["consistencia"] > 0 else previsto.new_zeros(())
        ),
    }
    total = sum(pesos[nome] * valor for nome, valor in termosDaLOSS.items())
    return total, termosDaLOSS


# ================================================================ DIAGNOSTICOS ================================================================
def baselineDePersistencia(loaderValidacao):
    criterio = t.nn.MSELoss()
    lossAcumulada = 0.0
    for janela, _, alvo in loaderValidacao:
        predPersistencia = janela[:, -1, :]
        lossAcumulada += criterio(predPersistencia, alvo).item()
    return lossAcumulada / len(loaderValidacao)

def horizontePrevisibilidade(real, previsto):
    escalaReferencia = np.linalg.norm(real - real.mean(axis=0), axis=1).mean()
    erro = np.linalg.norm(real - previsto, axis=1)
    acimaDoLimiar = np.where(erro > escalaReferencia)[0]
    horizonte = acimaDoLimiar[0] if len(acimaDoLimiar) > 0 else len(erro)
    return horizonte, escalaReferencia

def probabilidadeTeacherForcing(epoca, nEpocas):
    proporcao = epoca / nEpocas
    if proporcao <= 0.3:
        return 1.0
    fracaoRestante = (proporcao - 0.3) / (1 - 0.3)
    return max(1.0 - fracaoRestante, 0.2)

def validacaoUmPasso(modelo, loader, device):
    modelo.eval()
    criterio = t.nn.MSELoss()
    acumulador = 0.0
    with t.no_grad():
        for janela, massa, alvo in loader:
            janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)
            acumulador += criterio(modelo(janela, massa), alvo).item()
    return acumulador / len(loader)


# ================================================================ AVALIAÇÃO ================================================================
def rolloutAutoregressivo(modelo, trajetoriaNormalizada, massaNormalizada, tamanhoJanela, nPassos, device):
    modelo.eval()
    janelaAtual = trajetoriaNormalizada[:tamanhoJanela].to(device).unsqueeze(0)
    massa = massaNormalizada.to(device).unsqueeze(0)
    previsoes = []
    with t.no_grad():
        for _ in range(nPassos):
            proximo = modelo(janelaAtual, massa)
            previsoes.append(proximo.squeeze(0).cpu().numpy())
            janelaAtual = t.cat([janelaAtual[:, 1:, :], proximo.unsqueeze(1)], dim=1)
    return np.array(previsoes)

def avaliarRollout(modelo, trajetoriasValidacao, massasValidacao, tamanhoJanela, amostragem, media, desvio, mediaMassa, desvioMassa, device, nTrajetoriasMax=12, nPassosDoRollout=100):
    modelo.eval()
    nTraj = min(len(trajetoriasValidacao), nTrajetoriasMax)
    fig, eixos = plt.subplots(4, 3, figsize=(18, 20))
    eixos = eixos.flatten()
    horizontes, errosMedios = [], []
    for idx in range(nTraj):
        trajReal = trajetoriasValidacao[idx][::amostragem]
        massaReal = massasValidacao[idx]
        nPassos = min(nPassosDoRollout, len(trajReal) - tamanhoJanela)
        if nPassos <= 0:
            continue
        trajNorm = normalizar(trajReal, media, desvio)
        massaNorm = normalizar(massaReal, mediaMassa, desvioMassa)
        previstoNorm = rolloutAutoregressivo(modelo, trajNorm, massaNorm, tamanhoJanela, nPassos, device)
        previstoReal = desnormalizar(previstoNorm, media, desvio).numpy()
        alvoReal = trajReal[tamanhoJanela:tamanhoJanela + nPassos]
        erro = np.linalg.norm(previstoReal - alvoReal, axis=1)
        erroMedio = erro.mean()
        errosMedios.append(erroMedio)
        horizonte, escalaRef = horizontePrevisibilidade(alvoReal, previstoReal)
        horizontes.append(horizonte)
        ax = eixos[idx]
        passos = np.arange(nPassos)
        ax.plot(passos, alvoReal[:, 0], label="Real (corpo 1, X)")
        ax.plot(passos, previstoReal[:, 0], "--", label="Predito (corpo 1, X)")
        ax.set_title(f"Traj {idx} | erro médio: {erroMedio:.3f} | horiz.: {horizonte}")
        ax.set_xlabel("Passo")
        ax.set_ylabel("Posição X")
        if idx == 0:
            ax.legend()
        ax.grid(True)
    for j in range(nTraj, len(eixos)):
        eixos[j].axis("off")
    plt.suptitle(f"Rollout autoregressivo — validação (mostrando {nTraj} de {len(trajetoriasValidacao)} trajetórias)")
    plt.tight_layout()
    plt.savefig("rollout_validacao.png", dpi=120)
    plt.close()
    print(f"\nErro médio entre trajetórias ({nTraj} avaliadas): {np.mean(errosMedios):.4f}")
    print(f"Desvio do erro médio entre trajetórias: {np.std(errosMedios):.4f}")
    print(f"Horizonte de previsibilidade — média: {np.mean(horizontes):.2f} | mín/máx: {np.min(horizontes)}/{np.max(horizontes)}")
    return horizontes, errosMedios

@t.no_grad()
def metricasUmPasso(modelo, loader, device, media, desvio, mediaMassa, desvioMassa, escalaDelta):
    modelo.eval()
    acc = {"razaoPersist": 0., "mseNorm": 0., "errPos": 0., "errVel": 0., "dE": 0., "dP": 0., "dL": 0.}
    n = 0
    posIdx = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    velIdx = [3, 4, 5, 9, 10, 11, 15, 16, 17]
    for janela, massa, alvo in loader:
        janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)
        prev = modelo(janela, massa)
        ult = janela[:, -1, :]
        acc["mseNorm"] += t.mean((prev - alvo) ** 2).item()
        acc["razaoPersist"] += (t.mean(((prev - alvo) / escalaDelta) ** 2) / t.mean(((ult - alvo) / escalaDelta) ** 2)).item()
        err = ((prev - alvo) / escalaDelta) ** 2
        acc["errPos"] += err[:, posIdx].mean().item()
        acc["errVel"] += err[:, velIdx].mean().item()
        P, R, M = (desnormalizar(x, media, desvio) for x in (prev, alvo, ult))
        m = desnormalizar(massa, mediaMassa, desvioMassa)
        H0 = calcularHamiltoniano(M, m)
        acc["dE"] += ((calcularHamiltoniano(P, m) - H0).abs() / H0.abs()).mean().item()
        acc["dP"] += lossMomentoLinear(P, R, m, escalaMomLin).mean().item()
        acc["dL"] += lossMomentoAngular(P, R, m, escalaMomAng).mean().item()
        n += 1
    return {k: v / n for k, v in acc.items()}


# ================================================================ TREINO ================================================================
if __name__ == "__main__":
    N_EPOCAS = 40
    TAMANHO_JANELA = 10
    D_MODEL = 256
    N_HEADS = 8
    N_CAMADAS = 8
    DIM_FEEDFORWARD = 256
    DROPOUT = 0.0
    HORIZONTE_MAX = 20
    SUBPASSOS_POR_INFO = 50
    AMOSTRAGEM = 1
    DT_EFETIVO = DT_3C * SALVAR_A_CADA_3C

    PESOS_LOSS = {
        "estado": 50.0,
        "energia": 60.0,
        "momentoLinear": 20.0,
        "momentoAngular": 20.0,
        "consistencia": 0.0,
    }

    pastaLosses = "evolucaoLosses_resid.txt"
    pastaMetricas = "metricas_resid.txt"

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f"Rodando em: {device}")
    print(f"Hipótese A: Yoshida n={N_YOSHIDA_BASE} + resíduo (escalaResid={FRACAO_ESCALA_RESID}*escalaDelta)")
    print(f"Checkpoints: {PREFIXO_CKPT}*.pt (não sobrescreve checkpointEpoca*.pt)")

    trajetorias, massas = carregarTrajetorias("simulacoesArtificiais/simulacoes3C")
    nValidacao = max(1, len(trajetorias) // 5)
    trajetoriasTreino, trajetoriasValidacao = trajetorias[nValidacao:], trajetorias[:nValidacao]
    massasTreino, massasValidacao = massas[nValidacao:], massas[:nValidacao]
    print("Total de trajetórias carregadas:", len(trajetorias))
    print("Trajetórias de treino:", len(trajetoriasTreino))
    print("Trajetórias de validação:", len(trajetoriasValidacao))

    media, desvio = calcularNormalizacao(trajetoriasTreino)
    mediaMassa = np.mean(massasTreino, axis=0)
    desvioMassa = np.std(massasTreino, axis=0)
    desvioMassa = np.where(desvioMassa < 1e-8, 1.0, desvioMassa)

    escalaDelta = calcularEscalaDelta(trajetoriasTreino, AMOSTRAGEM, media, desvio).to(device)
    escalaResid = (FRACAO_ESCALA_RESID * escalaDelta).clamp_min(1e-8)
    desvioT = t.as_tensor(desvio, dtype=t.float32, device=device)
    mediaT = t.as_tensor(media, dtype=t.float32, device=device)
    mediaMassaT = t.as_tensor(mediaMassa, dtype=t.float32, device=device)
    desvioMassaT = t.as_tensor(desvioMassa, dtype=t.float32, device=device)

    escalaEnergia, escalaMomLin, escalaMomAng = calcularEscalasFisicas(trajetoriasTreino, massasTreino, AMOSTRAGEM)
    escalaEnergia, escalaMomLin, escalaMomAng = escalaEnergia.to(device), escalaMomLin.to(device), escalaMomAng.to(device)

    datasetTreino = JanelaComHorizonte(trajetoriasTreino, massasTreino, TAMANHO_JANELA, HORIZONTE_MAX, AMOSTRAGEM, media, desvio, mediaMassa, desvioMassa)
    datasetValidacao = JanelaProximoEstado(trajetoriasValidacao, massasValidacao, TAMANHO_JANELA, AMOSTRAGEM, media, desvio, mediaMassa, desvioMassa)
    loaderTreino = DataLoader(datasetTreino, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    loaderValidacao = DataLoader(datasetValidacao, batch_size=256, shuffle=False)
    datasetTreinoUmPasso = JanelaProximoEstado(trajetoriasTreino, massasTreino, TAMANHO_JANELA, AMOSTRAGEM, media, desvio, mediaMassa, desvioMassa)
    loaderTreinoUmPasso = DataLoader(datasetTreinoUmPasso, batch_size=256, shuffle=False)

    lossPersistenciaRef = baselineDePersistencia(loaderValidacao)
    print(f"Baseline de persistência (val, sem rede nenhuma): {lossPersistenciaRef:.6f}")

    modelo = RedeTransformer(
        dModel=D_MODEL, nHeads=N_HEADS, nCamadas=N_CAMADAS,
        dimFeedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
        escalaDelta=escalaDelta, escalaResid=escalaResid,
        media=mediaT, desvio=desvioT, mediaMassa=mediaMassaT, desvioMassa=desvioMassaT,
        dtEfetivo=DT_EFETIVO,
    ).to(device)
    otimizador = t.optim.Adam(modelo.parameters(), lr=1e-4)
    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=N_EPOCAS)

    ultimoCheckpoint, epocaInicial = acharUltimoCheckpoint(".", PREFIXO_CKPT)
    if ultimoCheckpoint is not None:
        print(f"Retomando checkpoint: {ultimoCheckpoint.name} (época {epocaInicial})")
        ckpt = t.load(ultimoCheckpoint, map_location=device, weights_only=False)
        modelo.load_state_dict(ckpt["modelo"])
        otimizador.load_state_dict(ckpt["otimizador"])
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        print("Nenhum checkpoint residual encontrado --> treinando do zero")

    NOMES = list(PESOS_LOSS.keys())
    for epoca in range(epocaInicial, N_EPOCAS):
        modelo.train()
        horizonteDaEpocaAtual = horizonteAtual(epoca, HORIZONTE_MAX, N_EPOCAS)
        lossTreinoAcumulada = 0.0
        lossesIndividuaisAcumuladas = t.zeros(5, device=device)
        nChamadas = 0
        probTF = probabilidadeTeacherForcing(epoca, N_EPOCAS)

        for janela, massa, futuros in tqdm(loaderTreino, desc=f"Época {epoca}   (horizonte da época = {horizonteDaEpocaAtual})"):
            janela, massa, futuros = janela.to(device), massa.to(device), futuros.to(device)
            otimizador.zero_grad()
            somaLossesParaEvitarRunMemory = 0.0
            B = janela.shape[0]
            janelaAtual = janela
            for h in range(horizonteDaEpocaAtual):
                alvo = futuros[:, h, :]
                ultimoEstado = janelaAtual[:, -1, :]
                previsto = modelo(janelaAtual, massa)
                estadoFis = desnormalizar(ultimoEstado, media, desvio)
                massaFis = desnormalizar(massa, mediaMassa, desvioMassa)
                with t.no_grad():
                    alvoFis = integrar(estadoFis, DT_EFETIVO, massaFis[..., 0], massaFis[..., 1], massaFis[..., 2], SUBPASSOS_POR_INFO)
                loss, termos = lossTotal(
                    desnormalizar(previsto, media, desvio), alvoFis, estadoFis, massaFis,
                    DT_EFETIVO, PESOS_LOSS, SUBPASSOS_POR_INFO,
                    escalaResid * desvioT, escalaEnergia, escalaMomLin, escalaMomAng,
                )
                (loss / horizonteDaEpocaAtual).backward()
                somaLossesParaEvitarRunMemory += loss.item()
                lossesIndividuaisAcumuladas += t.stack([termos[x].detach().mean() for x in NOMES])
                nChamadas += 1
                usarReal = (t.rand(B, device=device) < probTF).unsqueeze(1)
                proximoEstado = t.where(usarReal, alvo, previsto.detach())
                janelaAtual = t.cat([janelaAtual[:, 1:, :], proximoEstado.unsqueeze(1)], dim=1)

            t.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            otimizador.step()
            lossFinal = somaLossesParaEvitarRunMemory / horizonteDaEpocaAtual
            lossTreinoAcumulada += lossFinal

        lossTreinoAcumulada /= len(loaderTreino)
        est, ener, mlin, mang, cons = (lossesIndividuaisAcumuladas / nChamadas).cpu().tolist()

        metricasTreino = metricasUmPasso(modelo, loaderTreinoUmPasso, device, media, desvio, mediaMassa, desvioMassa, escalaDelta)
        metricasValidacao = metricasUmPasso(modelo, loaderValidacao, device, media, desvio, mediaMassa, desvioMassa, escalaDelta)
        modelo.train()

        with open(pastaMetricas, "a") as file:
            file.write(
                f"EPOCA {epoca} - " f"TREINO razaoPersist={metricasTreino['razaoPersist']:.6f} mseNorm={metricasTreino['mseNorm']:.6f} "
                f"errPos {metricasTreino['errPos']:.6f} - " f"errVel={metricasTreino['errVel']:.6f} "
                f"dE={metricasTreino['dE']:.6f} dP={metricasTreino['dP']:.6e} dL={metricasTreino['dL']:.6e} - "
                f"\n EPOCA {epoca} VAL    razaoPersist={metricasValidacao['razaoPersist']:.6f} mseNorm={metricasValidacao['mseNorm']:.6f} "
                f"errPos {metricasValidacao['errPos']:.6f} - " f"errVel={metricasValidacao['errVel']:.6f} "
                f"dE={metricasValidacao['dE']:.6f} dP={metricasValidacao['dP']:.6e} dL={metricasValidacao['dL']:.6e}\n"
            )
        with open(pastaLosses, "a") as file:
            file.write(
                f"EPOCA {epoca} ({horizonteDaEpocaAtual}) TF {probTF:.3f} - TOT {lossTreinoAcumulada:.8f} "
                f"EST {(est * PESOS_LOSS['estado']):.8f} ENER {(ener * PESOS_LOSS['energia']):.8f} "
                f"MOMlIN {(mlin * PESOS_LOSS['momentoLinear']):.8f} MOMaNG {(mang * PESOS_LOSS['momentoAngular']):.8f} "
                f"CONSIST {(cons * PESOS_LOSS['consistencia']):.8f}\n"
            )
        print(f"Época{epoca:3d} | TF {probTF:.2f} | h={horizonteDaEpocaAtual} | loss: {lossTreinoAcumulada:.8f}")

        scheduler.step()
        t.save(
            {"epoca": epoca, "modelo": modelo.state_dict(), "otimizador": otimizador.state_dict(), "scheduler": scheduler.state_dict()},
            f"{PREFIXO_CKPT}{epoca:03d}.pt",
        )

    t.save({
        "state_dict": modelo.state_dict(),
        "media": media, "desvio": desvio,
        "mediaMassa": mediaMassa, "desvioMassa": desvioMassa,
    }, "modelo3corpos_resid.pt")
    print("Modelo salvo em modelo3corpos_resid.pt")
    print("\n--- Rollout autoregressivo ---")
    avaliarRollout(
        modelo, trajetoriasValidacao, massasValidacao, TAMANHO_JANELA, AMOSTRAGEM,
        media, desvio, mediaMassa, desvioMassa, device, nTrajetoriasMax=12, nPassosDoRollout=100,
    )
