#!/usr/bin/env python3
"""
diagnostico.py -- por que o rollout da rede de 3 corpos perde a fase?

Uso (na mesma pasta do saturn2.py, com os checkpointEpoca*.pt):
    python3 diagnostico.py
    python3 diagnostico.py --checkpoints 22 25 31 39 --amostras 256 --passos 100

Saidas (pasta diagnostico_resultados/):
    relatorio.txt          tudo que e impresso no terminal
    curvas.png             erro por passo, horizonte por epoca, 1 passo, sensibilidade
    trajetorias.png        corpo 1 (X) real vs previsto, melhor checkpoint
    energia_trajetorias.png  H(t)/H0-1 real vs previsto, melhor checkpoint
    falhas.png             horizonte vs caracteristicas da trajetoria
    resultados.npz         arrays brutos para analise posterior

IMPORTANTE: os hiperparametros abaixo precisam ser iguais aos do treino.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch as t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import *
import saturn2 as S
import saturn3b as S3

# ======================= CONFIG (igual ao treino) =======================
PASTA_DADOS = "simulacoesArtificiais/simulacoes2C"
TAMANHO_JANELA = 10
D_MODEL = 256
N_HEADS = 8
N_CAMADAS = 8
DIM_FEEDFORWARD = 256
DROPOUT = 0.0
HORIZONTE_TREINO = 20
SUBPASSOS = 50
AMOSTRAGEM = 1
DT_EFETIVO = DT_2C * SALVAR_A_CADA_2C
POS_IDX = [0, 1, 2, 6, 7, 8]
VEL_IDX = [3, 4, 5, 9, 10, 11]
PASSOS_TABELA = [1, 5, 10, 20, 40, 60, 80, 100, 150, 200, 300, 500, 700, 1000]
CKPTS_PADRAO = [14, 20, 22, 25, 28, 31, 34, 37, 39]
INF = float("inf")


# ======================= utilidades =======================
class Tee:
    def __init__(self, caminho):
        self.arq = open(caminho, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.arq.write(s)

    def flush(self):
        self.out.flush()
        self.arq.flush()


class Obj:
    pass


def secao(titulo):
    print("\n" + "=" * 110)
    print(titulo)
    print("=" * 110)


def norm(x, d):
    return (x - d.media) / d.desvio


def desnorm(x, d):
    return x * d.desvio + d.media


def limpa(x):
    return t.nan_to_num(x, nan=INF, posinf=INF)


def npy(x):
    return x.detach().cpu().numpy().astype(np.float64)


def fin(a):
    return np.where(np.isfinite(a), a, np.nan)


def med(a, axis=0):
    return np.nanmedian(a, axis=axis)


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok]))
    rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def distanciasPares(estado):
    return (estado[...,6:9]-estado[...,0:3]).norm(dim=-1).unsqueeze(-1)

def primeiroAcima(err, lim):
    """numero de passos sobrevividos antes do primeiro erro > limiar (T se nunca passa)"""
    acima = err > lim
    idx = acima.to(t.int64).argmax(dim=1)
    return t.where(acima.any(dim=1), idx, t.full_like(idx, err.shape[1]))


# ======================= dados / modelo =======================
def calcularNormalizacao2C(trajetorias):
    todos = np.concatenate([tr[:, :12] for tr in trajetorias]).astype(np.float64)
    media = todos.mean(axis=0)
    desvio = todos.std(axis=0)
    desvio = np.where(desvio < 1e-8, 1.0, desvio)
    return media, desvio


def calcularEscalaDelta2C(trajetorias, amostragem, media, desvio):
    """desvio padrao (em espaco normalizado) da diferenca entre estados consecutivos"""
    deltas = []
    for tr in trajetorias:
        x = (tr[:, :12] - media) / desvio
        deltas.append(x[amostragem:] - x[:-amostragem])
    dd = np.concatenate(deltas)
    sd = dd.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return t.as_tensor(sd, dtype=t.float32)


def calcularEscalasFisicas2C(trajetorias, massas, amostragem):
    """escala tipica da variacao de energia, momento linear e momento angular entre passos"""
    dE, dP, dL = [], [], []
    for tr, m in zip(trajetorias, massas):
        e = t.as_tensor(tr[:, :12], dtype=t.float32)
        mt = t.as_tensor(m, dtype=t.float32).unsqueeze(0).expand(len(e), 2)
        H = S3.calcularHamiltoniano(e, mt)
        P = S3.calcularMomentoLinearVetor(e, mt)
        L = S3.calcularMomentoAngular(e, mt)
        dE.append((H[amostragem:] - H[:-amostragem]).abs())
        dP.append((P[amostragem:] - P[:-amostragem]).norm(dim=-1))
        dL.append((L[amostragem:] - L[:-amostragem]).norm(dim=-1))
    esc = lambda x: t.cat(x).std().clamp_min(1e-12)
    return esc(dE), esc(dP), esc(dL)

def prepararDados(device):
    d = Obj()
    trajetorias, massas = S3.carregarTrajetorias(PASTA_DADOS)
    nVal = max(1, len(trajetorias) // 5)
    d.trajVal, d.massasVal = trajetorias[:nVal], massas[:nVal]
    trajTreino, massasTreino = trajetorias[nVal:], massas[nVal:]
    media, desvio = calcularNormalizacao2C(trajTreino)
    mediaMassa = np.mean(massasTreino, axis=0)
    desvioMassa = np.std(massasTreino, axis=0)
    desvioMassa = np.where(desvioMassa < 1e-8, 1.0, desvioMassa)
    tt = lambda x: t.as_tensor(x, dtype=t.float32, device=device)
    d.media, d.desvio = tt(media), tt(desvio)
    d.mediaMassa, d.desvioMassa = tt(mediaMassa), tt(desvioMassa)
    d.escalaDelta = calcularEscalaDelta2C(trajTreino, AMOSTRAGEM, media, desvio).to(device)
    eE, eP, eL = calcularEscalasFisicas2C(trajTreino, massasTreino, AMOSTRAGEM)
    d.escalaEnergia, d.escalaMomLin, d.escalaMomAng = eE.to(device), eP.to(device), eL.to(device)
    return d


def montarConjunto(d, nAmostras, passos, modo, seed, device):
    comp = TAMANHO_JANELA + passos
    validos = [i for i, tr in enumerate(d.trajVal) if len(tr) >= comp]
    if not validos:
        raise RuntimeError(f"nenhuma trajetoria de validacao tem {comp} passos")
    rng = np.random.default_rng(seed)
    if modo == "zero":
        ids = validos[:nAmostras]
        starts = [0] * len(ids)
    else:
        ids = [int(i) for i in rng.choice(validos, size=nAmostras, replace=True)]
        starts = [int(rng.integers(0, len(d.trajVal[i]) - comp + 1)) for i in ids]
    fis = np.stack([d.trajVal[i][a:a + comp] for i, a in zip(ids, starts)]).astype(np.float32)
    mas = np.stack([d.massasVal[i] for i in ids]).astype(np.float32)
    c = Obj()
    c.ids, c.starts = np.array(ids), np.array(starts)
    c.fis = t.as_tensor(fis, device=device)
    c.massaFis = t.as_tensor(mas, device=device)
    c.norm = norm(c.fis, d)
    c.massaNorm = (c.massaFis - d.mediaMassa) / d.desvioMassa
    c.janela = c.norm[:, :TAMANHO_JANELA]
    c.alvoNorm = c.norm[:, TAMANHO_JANELA:]
    c.alvoFis = c.fis[:, TAMANHO_JANELA:]
    c.ultimoFis = c.fis[:, TAMANHO_JANELA - 1]
    return c


def listarCheckpoints(pasta, selecao, prefixo="checkpointEpoca"):
    mapa = {}
    for p in sorted(Path(pasta).glob(prefixo + "*.pt")):
        mapa[int(re.search(r"(\d+)", p.stem).group(1))] = p
    if not mapa:
        raise RuntimeError(f"nenhum {prefixo}*.pt encontrado em {pasta}")
    eps = selecao if selecao else [e for e in CKPTS_PADRAO if e in mapa]
    if not eps:
        eps = sorted(mapa)
    return [(e, mapa[e]) for e in sorted(eps) if e in mapa]


def carregarModelo(caminho, device):
    ck = t.load(caminho, map_location=device, weights_only=False)
    if "config" in ck:                      # checkpoint do saturn3 (janela 1, tokens por corpo)
        import saturn3b as S3
        modelo = S3.RedeCorpos(**ck["config"]).to(device)
    else:                                   # checkpoints antigos do saturn2
        modelo = S.RedeTransformer(dModel=D_MODEL, nHeads=N_HEADS, nCamadas=N_CAMADAS,
                                   dimFeedforward=DIM_FEEDFORWARD, dropout=DROPOUT).to(device)
    modelo.load_state_dict(ck["modelo"] if "modelo" in ck else ck["state_dict"])
    modelo.eval()
    return modelo


# ======================= rollouts =======================
PROJETAR_D = None      # objeto de dados (d) quando alguma projecao esta ativa
PROJETAR_TUDO = False  # False: so energia | True: momento linear + momento angular + energia


def _vel3(P):
    return P[:, VEL_IDX].reshape(-1, 3, 3)      # (B, corpo, xyz)


def _pos3(P):
    return P[:, POS_IDX].reshape(-1, 3, 3)


def projetarEnergia(prevNorm, H0, massaNorm, d, limites=(0.64, 1.56)):
    """Mantem as posicoes previstas e reescala as velocidades para que H(previsto) = H0."""
    P = desnorm(prevNorm, d)
    m = massaNorm * d.desvioMassa + d.mediaMassa
    P0 = P.clone()
    P0[:, VEL_IDX] = 0
    U = S3.calcularHamiltoniano(P0, m)                   # so a potencial
    KEprev = S3.calcularHamiltoniano(P, m) - U           # cinetica prevista
    KEalvo = H0 - U                                     # cinetica que conserva a energia
    f = t.sqrt((KEalvo / KEprev.clamp_min(1e-8)).clamp(*limites))
    P[:, VEL_IDX] = P[:, VEL_IDX] * f[:, None]
    return norm(P, d)


def projetarConservacao(prevNorm, alvos, massaNorm, d, iteracoes=3):
    """Corrige as velocidades previstas para conservar momento linear, momento angular e energia.
    Momentos: menor correcao (em norma ponderada por massa) dada por dv_i = a + w x r_i, com (a, w) resolvendo um
    sistema 6x6 (testado em numpy: erro ~1e-15). Energia: reescala a velocidade relativa ao centro de massa."""
    H0, P0, L0 = alvos
    P = desnorm(prevNorm, d)
    m = massaNorm * d.desvioMassa + d.mediaMassa            # (B,3)
    M = m.sum(1)                                            # (B,)
    r = _pos3(P)                                            # (B,3,3)
    S_ = (m[..., None] * r).sum(1)                          # (B,3)
    eye = t.eye(3, device=P.device, dtype=P.dtype)
    r2_ = (r * r).sum(-1)                                   # (B,3)
    I = (m * r2_).sum(1)[:, None, None] * eye - (m[..., None, None] * r[..., :, None] * r[..., None, :]).sum(1)
    Icm = I - ((S_ * S_).sum(1)[:, None, None] * eye - S_[:, :, None] * S_[:, None, :]) / M[:, None, None]
    Icm = Icm + 1e-8 * Icm.diagonal(dim1=-2, dim2=-1).sum(-1)[:, None, None] * eye
    U = None
    for _ in range(iteracoes):
        v = _vel3(P)
        dP = P0 - (m[..., None] * v).sum(1)
        dL = L0 - (m[..., None] * t.cross(r, v, dim=-1)).sum(1)
        w = t.linalg.solve(Icm, (dL - t.cross(S_, dP, dim=-1) / M[:, None])[..., None])[..., 0]
        a = (dP - t.cross(w, S_, dim=-1)) / M[:, None]
        v = v + a[:, None, :] + t.cross(w[:, None, :].expand_as(r), r, dim=-1)
        # energia: reescala a velocidade relativa ao centro de massa
        Vcm = (m[..., None] * v).sum(1) / M[:, None]
        rel = v - Vcm[:, None, :]
        P[:, VEL_IDX] = v.reshape(-1, 9)
        if U is None:
            Pz = P.clone(); Pz[:, VEL_IDX] = 0
            U = S3.calcularHamiltoniano(Pz, m)
        KE = S3.calcularHamiltoniano(P, m) - U
        KEcm = 0.5 * M * (Vcm * Vcm).sum(-1)
        KErel = (KE - KEcm).clamp_min(1e-8)
        f = t.sqrt(((H0 - U - KEcm) / KErel).clamp(0.64, 1.56))
        P[:, VEL_IDX] = (Vcm[:, None, :] + f[:, None, None] * rel).reshape(-1, 9)
    return norm(P, d)


@t.no_grad()
def rolloutLote(modelo, janela, massaNorm, passos):
    atual = janela.clone()
    saida = []
    alvos = None
    if PROJETAR_D is not None:
        d = PROJETAR_D
        m0 = massaNorm * d.desvioMassa + d.mediaMassa
        E0 = desnorm(janela[:, -1], d)                                  # estado real inicial
        H0 = S3.calcularHamiltoniano(E0, m0)
        P0 = S3.calcularMomentoLinearVetor(E0, m0)
        L0 = S3.calcularMomentoAngular(E0, m0)
        alvos = (H0, P0, L0)
    for _ in range(passos):
        prox = modelo(atual, massaNorm)
        if alvos is not None:
            if PROJETAR_TUDO:
                prox = projetarConservacao(prox, alvos, massaNorm, PROJETAR_D)
            else:
                prox = projetarEnergia(prox, alvos[0], massaNorm, PROJETAR_D)
        saida.append(prox)
        atual = t.cat([atual[:, 1:], prox.unsqueeze(1)], dim=1)
    return t.stack(saida, dim=1)


@t.no_grad()
def rolloutIntegrador(estado0Fis, massaFis, passos, nSub, ruidoFis=None):
    est = estado0Fis.contiguous()
    m1, m2 = massaFis[:, 0].contiguous(), massaFis[:, 1].contiguous()
    saida = []
    for _ in range(passos):
        est = S3.integrar(est, DT_EFETIVO, m1, m2, nSub)
        if ruidoFis is not None:
            est = est + t.randn_like(est) * ruidoFis
        saida.append(est)
    return t.stack(saida, dim=1)


# ======================= metricas =======================
@t.no_grad()
def metricasRollout(c, prevNorm, d):
    prev = desnorm(prevNorm, d)
    real = c.alvoFis
    N, T, _ = real.shape
    dif = prev - real
    err18 = limpa(dif.norm(dim=-1))
    erroPos = limpa(dif[..., POS_IDX].norm(dim=-1))
    erroVel = limpa(dif[..., VEL_IDX].norm(dim=-1))
    ref = (real - real.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=1)   # mesma escala do horizontePrevisibilidade
    dist = distanciasPares(real)
    tam = dist.mean(dim=(1, 2))
    mm = c.massaFis[:, None, :].expand(N, T, 2)
    ult = c.ultimoFis
    H0 = S3.calcularHamiltoniano(ult, c.massaFis)
    den = H0.abs().clamp_min(1e-6)[:, None]
    derP = (S3.calcularHamiltoniano(prev, mm) - H0[:, None]) / den
    derR = (S3.calcularHamiltoniano(real, mm) - H0[:, None]) / den
    P0 = S3.calcularMomentoLinearVetor(ult, c.massaFis)
    L0 = S3.calcularMomentoAngular(ult, c.massaFis)
    dP = (S3.calcularMomentoLinearVetor(prev, mm) - P0[:, None]).norm(dim=-1) / d.escalaMomLin
    dL = (S3.calcularMomentoAngular(prev, mm) - L0[:, None]).norm(dim=-1) / d.escalaMomAng
    distP = distanciasPares(prev)
    escape = limpa(distP[:, -1].amax(dim=-1)) > 2.0 * dist.amax(dim=(1, 2))
    return dict(
        erroPos=npy(erroPos), erroVel=npy(erroVel), err18=npy(err18),
        derP=npy(derP), derR=npy(derR), dP=npy(dP), dL=npy(dL),
        h_orig=npy(primeiroAcima(err18, ref[:, None])),
        h1=npy(primeiroAcima(erroPos, 0.01 * tam[:, None])),
        h10=npy(primeiroAcima(erroPos, 0.10 * tam[:, None])),
        h50=npy(primeiroAcima(erroPos, 0.50 * tam[:, None])),
        tam=npy(tam), escape=npy(escape.float()),
    )


@t.no_grad()
def diagnosticoUmPasso(modelo, c, d):
    N = c.janela.shape[0]
    prev = t.cat([modelo(c.janela[a:a + 512], c.massaNorm[a:a + 512]) for a in range(0, N, 512)])
    alvo = c.alvoNorm[:, 0]
    ult = c.janela[:, -1]
    sd = d.escalaDelta
    e = (prev - alvo) / sd
    ep = (ult - alvo) / sd
    r = {"rms_comp": e.pow(2).mean(0).sqrt()}
    r["rms_pos"] = e[:, POS_IDX].pow(2).mean().sqrt().item()
    r["rms_vel"] = e[:, VEL_IDX].pow(2).mean().sqrt().item()
    r["pers_pos"] = ep[:, POS_IDX].pow(2).mean().sqrt().item()
    r["pers_vel"] = ep[:, VEL_IDX].pow(2).mean().sqrt().item()
    r["vies_pos"] = e[:, POS_IDX].mean().item()
    r["vies_vel"] = e[:, VEL_IDX].mean().item()
    pd, td = (prev - ult) / sd, (alvo - ult) / sd
    for nome, idx in (("pos", POS_IDX), ("vel", VEL_IDX)):
        a_, b_ = pd[:, idx].flatten(), td[:, idx].flatten()
        r["ganho_" + nome] = ((a_ * b_).sum() / (b_ * b_).sum()).item()
        r["corr_" + nome] = t.corrcoef(t.stack([a_, b_]))[0, 1].item()
    Pf, Rf, Uf = desnorm(prev, d), desnorm(alvo, d), desnorm(ult, d)
    m = c.massaFis
    Hp, Hr, Hu = S3.calcularHamiltoniano(Pf, m), S3.calcularHamiltoniano(Rf, m), S3.calcularHamiltoniano(Uf, m)
    den = Hu.abs().clamp_min(1e-6)
    r["dE_vies"] = ((Hp - Hr) / den).mean().item()
    r["dE_med"] = ((Hp - Hr) / den).abs().median().item()
    r["dE_rms"] = ((Hp - Hr) / den).pow(2).mean().sqrt().item()
    r["dE_dataset_rms"] = ((Hr - Hu) / den).pow(2).mean().sqrt().item()
    return r


@t.no_grad()
def consistenciaAlvo(c, d, ns):
    """Yoshida a partir do ultimo estado real vs proximo estado real do dataset (unidades de escalaDelta)."""
    ult = c.ultimoFis.contiguous()
    m = c.massaFis
    sdFis = d.escalaDelta * d.desvio
    res = {}
    for n in ns:
        out = S3._integrarEager(ult, DT_EFETIVO, m[:, 0], m[:, 1], n)
        e = (out - c.alvoFis[:, 0]) / sdFis
        res[n] = dict(rms_pos=e[:, POS_IDX].pow(2).mean().sqrt().item(),
                      rms_vel=e[:, VEL_IDX].pow(2).mean().sqrt().item(),
                      med_pos=e[:, POS_IDX].abs().median().item(),
                      med_vel=e[:, VEL_IDX].abs().median().item())
    return res


@t.no_grad()
def sensibilidadeRede(modelo, c, d, delta, passos):
    sd = d.escalaDelta
    a = rolloutLote(modelo, c.janela, c.massaNorm, passos)
    b = rolloutLote(modelo, c.janela + delta[:, None, :], c.massaNorm, passos)
    den = (delta / sd).norm(dim=-1)[:, None]
    return npy(((b - a) / sd).norm(dim=-1) / den)


@t.no_grad()
def sensibilidadeVerdadeira(c, d, delta, passos):
    base = rolloutIntegrador(c.ultimoFis, c.massaFis, passos, SUBPASSOS)
    pert = rolloutIntegrador(c.ultimoFis + delta * d.desvio, c.massaFis, passos, SUBPASSOS)
    den = (delta / d.escalaDelta).norm(dim=-1)[:, None]
    return npy((((pert - base) / d.desvio) / d.escalaDelta).norm(dim=-1) / den)


def ajusteCrescimento(curva, t0=3):
    T = len(curva)
    x = np.arange(t0, T + 1)
    y = np.log(np.maximum(curva[x - 1], 1e-30))
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return None
    p = np.polyfit(np.log(x), y, 1)
    e = np.polyfit(x, y, 1)
    return dict(expoente=p[0], r2_pot=r2(y, np.polyval(p, np.log(x))),
                taxa=e[0], dobra=(np.log(2) / e[0] if e[0] > 0 else INF),
                r2_exp=r2(y, np.polyval(e, x)))


def caracteristicas(c):
    real = c.alvoFis
    dist = distanciasPares(real)
    mm = c.massaFis
    return {
        "dist_min": npy(dist.amin(dim=(1, 2))),
        "tamanho": npy(dist.mean(dim=(1, 2))),
        "energia_H0": npy(S3.calcularHamiltoniano(c.ultimoFis, mm)),
        "mom_angular": npy(S3.calcularMomentoAngular(c.ultimoFis, mm).norm(dim=-1)),
        "razao_massas": npy(mm.amax(dim=1) / mm.amin(dim=1)),
        "vel_tipica": npy(real[..., VEL_IDX].norm(dim=-1).mean(dim=1)),
        "ini_na_traj": c.starts.astype(np.float64),
    }


# ======================= impressao =======================
def imprimirCurvas(titulo, curvas):
    print(f"\n{titulo}")
    nomes = list(curvas)
    T = len(next(iter(curvas.values())))
    print("passo " + " ".join(f"{n[:12]:>12}" for n in nomes))
    for s in [s for s in PASSOS_TABELA if s <= T]:
        print(f"{s:5d} " + " ".join(f"{curvas[n][s - 1]:12.3e}" for n in nomes))


def resumoBase(nome, m):
    return (f"{nome:<26} h_orig med/media={np.median(m['h_orig']):6.1f}/{np.mean(m['h_orig']):6.1f}  "
            f"h1%={np.median(m['h1']):5.1f} h10%={np.median(m['h10']):5.1f} h50%={np.median(m['h50']):5.1f}  "
            f"errPos@10={med(m['erroPos'])[9]:.2e} @T={med(m['erroPos'])[-1]:.2e}")


def tabelaQuartis(nome, x, h):
    ok = np.isfinite(x)
    if ok.sum() < 8:
        return
    q = np.quantile(x[ok], [0, .25, .5, .75, 1.0])
    print(f"  por quartil de {nome}:")
    for i in range(4):
        sel = ok & (x >= q[i]) & (x <= q[i + 1])
        if sel.sum():
            print(f"    Q{i + 1} [{q[i]:.3g}, {q[i + 1]:.3g}]  n={sel.sum():3d}  mediana h={np.median(h[sel]):6.1f}")


def linhaResumo(ep, r, T, sensVerd):
    z, a, um = r["zero"], r["aleatorio"], r["um"]
    s = min(50, T)
    sr = med(r["sens"])[s - 1] / max(med(sensVerd)[s - 1], 1e-30)
    return (f"{ep:5d} | {np.median(z['h_orig']):5.1f} {np.median(a['h_orig']):5.1f} | "
            f"{np.median(a['h1']):4.1f} {np.median(a['h10']):5.1f} {np.median(a['h50']):5.1f} | "
            f"{med(a['erroPos'])[9]:8.2e} {med(a['erroPos'])[s - 1]:8.2e} | "
            f"{um['rms_pos']:7.4f} {um['rms_vel']:7.4f} | {um['ganho_pos']:6.3f} {um['ganho_vel']:6.3f} | "
            f"{med(a['derP'])[s - 1]:+9.2e} | {sr:7.2f} | {np.median(r['aleatorio_ruido']['h_orig']):5.1f}")


# ======================= figuras =======================
def figuraCurvas(res, base, eps, best, T, umRef, sensVerd, saida):
    cores = plt.cm.viridis(np.linspace(0, 1, len(eps)))
    x = np.arange(1, T + 1)
    fig, ax = plt.subplots(2, 3, figsize=(21, 11))
    with np.errstate(all="ignore"):
        for cor, ep in zip(cores, eps):
            m = res[ep]["aleatorio"]
            lw = 2.8 if ep == best else 1.0
            ax[0, 0].plot(x, med(m["erroPos"]), color=cor, lw=lw, label=f"ép {ep}")
            ax[0, 1].plot(x, med(m["erroVel"]), color=cor, lw=lw, label=f"ép {ep}")
            ax[1, 2].plot(x, med(res[ep]["sens"]), color=cor, lw=lw, label=f"ép {ep}")
        rb = res[best]
        for k, (chave, estilo, rot) in enumerate([(rb["aleatorio_ruido"], "k--", "ruído equivalente"),
                                                  *[(base[("aleatorio", n)], ":", f"Yoshida n={n}") for n in sorted({n for (_, n) in base})]]):
            ax[0, 0].plot(x, med(chave["erroPos"]), estilo, label=rot)
            ax[0, 1].plot(x, med(chave["erroVel"]), estilo, label=rot)
        ax[1, 2].plot(x, med(sensVerd), "k--", lw=2.2, label="sistema real (Yoshida)")
        for a_ in (ax[0, 0], ax[0, 1], ax[1, 2]):
            a_.set_yscale("log")
            a_.axvline(TAMANHO_JANELA, color="gray", ls="-.", lw=0.8)
            a_.axvline(HORIZONTE_TREINO, color="red", ls="-.", lw=0.8)
            a_.set_xlabel("passo à frente (cinza: janela toda prevista; vermelho: horizonte do treino)")
            a_.grid(True, alpha=.3)
        ax[0, 0].set_title("erro de posição (mediana, conjunto aleatório)")
        ax[0, 1].set_title("erro de velocidade (mediana)")
        ax[1, 2].set_title("sensibilidade: ||Δ saída|| / ||Δ entrada|| (rede vs física)")
        ax[0, 0].legend(fontsize=7, ncol=2)
        ax[1, 2].legend(fontsize=7, ncol=2)

        m = res[best]["aleatorio"]
        dp = fin(m["derP"])
        ax[0, 2].plot(x, np.nanmedian(dp, 0), "C3", label=f"previsto (ép {best})")
        ax[0, 2].fill_between(x, np.nanpercentile(dp, 10, 0), np.nanpercentile(dp, 90, 0), color="C3", alpha=.2)
        ax[0, 2].plot(x, np.nanmedian(fin(m["derR"]), 0), "k--", label="real (dataset)")
        ax[0, 2].set_title("deriva relativa de energia (H-H0)/|H0|, mediana e P10-P90")
        ax[0, 2].set_ylim(-0.5, 0.5)
        ax[0, 2].legend()
        ax[0, 2].grid(True, alpha=.3)

        ax[1, 0].plot(eps, [np.median(res[e]["zero"]["h_orig"]) for e in eps], "o-", label="h_orig (início=0)")
        ax[1, 0].plot(eps, [np.median(res[e]["aleatorio"]["h_orig"]) for e in eps], "o-", label="h_orig (aleatório)")
        ax[1, 0].plot(eps, [np.median(res[e]["aleatorio"]["h10"]) for e in eps], "s-", label="h erro<10% do tamanho")
        ax[1, 0].plot(eps, [np.median(res[e]["aleatorio_ruido"]["h_orig"]) for e in eps], "k--", label="ruído equivalente")
        ax[1, 0].set_title("horizonte (mediana de passos) por época")
        ax[1, 0].set_xlabel("época")
        ax[1, 0].legend(fontsize=8)
        ax[1, 0].grid(True, alpha=.3)

        ax[1, 1].plot(eps, [res[e]["um"]["rms_pos"] for e in eps], "o-", label="rms pos (1 passo)")
        ax[1, 1].plot(eps, [res[e]["um"]["rms_vel"] for e in eps], "o-", label="rms vel (1 passo)")
        ax[1, 1].set_yscale("log")
        ax[1, 1].set_title("erro de 1 passo (unid. escalaDelta); tracejado = Yoshida(50) vs dataset")
        if umRef:
            ax[1, 1].axhline(umRef[0], color="C0", ls="--")
            ax[1, 1].axhline(umRef[1], color="C1", ls="--")
        ax[1, 1].set_xlabel("época")
        ax[1, 1].legend(fontsize=8)
        ax[1, 1].grid(True, alpha=.3)
    plt.tight_layout()
    plt.savefig(saida / "curvas.png", dpi=110)
    plt.close()


def figurasTrajetorias(res, roll, conjZero, best, d, saida):
    prev = npy(desnorm(roll[(best, "zero")], d))
    real = npy(conjZero.alvoFis)
    m = res[best]["zero"]
    n = min(12, real.shape[0])
    for nome, arq in (("pos", "trajetorias.png"), ("energia", "energia_trajetorias.png")):
        fig, eixos = plt.subplots(4, 3, figsize=(18, 14))
        for i, ax in enumerate(eixos.flatten()):
            if i >= n:
                ax.axis("off")
                continue
            if nome == "pos":
                ax.plot(real[i, :, 0], label="real (corpo 1, X)")
                ax.plot(prev[i, :, 0], "--", label="previsto")
            else:
                ax.plot(m["derR"][i], label="real")
                ax.plot(np.clip(fin(m["derP"][i]), -2, 2), "--", label="previsto")
            ax.axvline(m["h_orig"][i], color="red", lw=0.7)
            ax.set_title(f"traj {conjZero.ids[i]} | h={m['h_orig'][i]:.0f}")
            ax.grid(True, alpha=.3)
            if i == 0:
                ax.legend()
        plt.suptitle(f"época {best}: " + ("posição" if nome == "pos" else "(H-H0)/|H0|") + " (linha vermelha = horizonte)")
        plt.tight_layout()
        plt.savefig(saida / arq, dpi=100)
        plt.close()


def figuraFalhas(c, m, saida, best):
    f = caracteristicas(c)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].scatter(f["dist_min"], m["h_orig"], s=10)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("menor distância entre corpos na janela real")
    ax[0].set_ylabel("horizonte (passos)")
    ax[1].scatter(f["vel_tipica"], m["h_orig"], s=10)
    ax[1].set_xlabel("velocidade típica")
    ax[2].hist(m["h_orig"], bins=30)
    ax[2].set_xlabel("horizonte (passos)")
    for a_ in ax:
        a_.grid(True, alpha=.3)
    plt.suptitle(f"época {best}, conjunto aleatório")
    plt.tight_layout()
    plt.savefig(saida / "falhas.png", dpi=110)
    plt.close()


# ======================= leitura automatica =======================
def leituraAutomatica(res, base, cons, best, T, sensVerd):
    L = []
    rb, um = res[best], res[best]["um"]
    redeRms = max(um["rms_pos"], um["rms_vel"])
    c50 = cons[SUBPASSOS]
    ratio = max(c50["rms_pos"], c50["rms_vel"]) / max(redeRms, 1e-12)
    nbest = min(cons, key=lambda n: cons[n]["rms_pos"] + cons[n]["rms_vel"])
    L.append(f"[alvo x dataset] Yoshida(n={SUBPASSOS}) difere do dataset em 1 passo por {ratio:.2f}x o erro da rede. "
             + ("ALERTA: o alvo do treino não reproduz o dataset com a precisão da rede; ela aprende uma dinâmica ligeiramente diferente da avaliada."
                if ratio > 0.5 else "OK: o alvo do treino reproduz o dataset com folga."))
    if nbest != SUBPASSOS and (cons[nbest]["rms_pos"] + cons[nbest]["rms_vel"]) < 0.5 * (c50["rms_pos"] + c50["rms_vel"]):
        L.append(f"[alvo x dataset] n={nbest} reproduz o dataset bem melhor que n={SUBPASSOS}; "
                 f"o dataset provavelmente foi gerado com outro passo interno (SALVAR_A_CADA_2C={SALVAR_A_CADA_2C}).")
    hf = np.median(base[("aleatorio", SUBPASSOS)]["h_orig"])
    hr = np.median(rb["aleatorio"]["h_orig"])
    L.append(f"[piso float32] Yoshida({SUBPASSOS}) em rollout a partir do estado real: horizonte mediano {hf:.0f} (rede: {hr:.0f}). "
             + ("O próprio integrador diverge do dataset; parte da defasagem não é culpa da rede." if hf < 0.8 * T else "O integrador acompanha o dataset ao longo de todo o rollout."))
    hn = np.median(rb["aleatorio_ruido"]["h_orig"])
    L.append(f"[ruído equivalente] física exata + ruído com a mesma magnitude do erro de 1 passo da rede: horizonte {hn:.0f} (rede: {hr:.0f}). "
             + ("A rede falha bem mais cedo que ruído branco equivalente: o erro dela é estruturado (viés), não só amplitude."
                if hr < 0.6 * hn else "Comparável a ruído equivalente: reduzir o erro de 1 passo é a alavanca principal."))
    s = min(50, T)
    sr = med(rb["sens"])[s - 1] / max(med(sensVerd)[s - 1], 1e-30)
    L.append(f"[sensibilidade] amplificação de perturbações no passo {s}: rede/física = {sr:.2f}. "
             + ("A rede amplifica perturbações mais que o sistema real (instabilidade própria)." if sr > 2 else
                "A rede amplifica perturbações de forma parecida com a física."))
    L.append(f"[ganho do passo] pos={um['ganho_pos']:.4f} vel={um['ganho_vel']:.4f} (ideal 1). "
             + ("Viés sistemático no tamanho do passo (erro de frequência/fase)." if max(abs(um['ganho_pos'] - 1), abs(um['ganho_vel'] - 1)) > 0.01 else "Sem viés relevante no tamanho do passo."))
    dE = med(rb["aleatorio"]["derP"])[s - 1]
    dR = med(rb["aleatorio"]["derR"])[s - 1]
    L.append(f"[energia] mediana de (H-H0)/|H0| no passo {s}: previsto {dE:+.3e}, dataset {dR:+.3e}. "
             + ("Deriva de energia significativa na rede." if abs(dE) > 0.05 else "Deriva de energia pequena."))
    fit = ajusteCrescimento(med(rb["aleatorio"]["erroPos"]))
    if fit:
        L.append(f"[crescimento do erro] expoente de lei de potência={fit['expoente']:.2f} (R2={fit['r2_pot']:.3f}); "
                 f"taxa exponencial={fit['taxa']:.3f}/passo (dobra a cada {fit['dobra']:.1f} passos, R2={fit['r2_exp']:.3f}). "
                 + ("Ajuste exponencial melhor: instabilidade/caos domina." if fit['r2_exp'] > fit['r2_pot'] + 0.02 else
                    "Lei de potência ajusta igual ou melhor: acúmulo sistemático de fase."))
    L.append("(heurísticas; confirme olhando as tabelas e figuras)")
    return L


# ======================= main =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", type=int, nargs="*", default=None)
    ap.add_argument("--pasta", default=".")
    ap.add_argument("--prefixo", default="checkpointEpoca", help="ex.: ft_checkpointEpoca para os checkpoints do fine-tuning")
    ap.add_argument("--amostras", type=int, default=256)
    ap.add_argument("--passos", type=int, default=100)
    ap.add_argument("--amostras_umpasso", type=int, default=2048)
    ap.add_argument("--eps", type=float, default=0.3, help="perturbação do teste de sensibilidade, em unidades de escalaDelta")
    ap.add_argument("--saida", default="diagnostico_resultados")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--projetar_tudo", action="store_true", help="projeta momento linear, momento angular e energia a cada passo")
    ap.add_argument("--projetar_energia", action="store_true", help="projeta a energia (velocidades reescaladas) a cada passo do rollout")
    args = ap.parse_args()

    saida = Path(args.saida)
    saida.mkdir(exist_ok=True)
    sys.stdout = Tee(saida / "relatorio.txt")
    t0 = time.time()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    t.manual_seed(args.seed)

    d = prepararDados(device)
    global PROJETAR_D, PROJETAR_TUDO
    if args.projetar_energia or args.projetar_tudo:
        PROJETAR_D = d
        PROJETAR_TUDO = args.projetar_tudo
        print("*** rollout COM projecao de " + ("momentos + energia" if PROJETAR_TUDO else "energia") + " ***")
    S.escalaDelta = d.escalaDelta     # a RedeTransformer le essa variavel global do saturn2
    maxlen = max(len(tr) for tr in d.trajVal)
    args.passos = min(args.passos, maxlen - TAMANHO_JANELA - 1)
    T = args.passos
    ckpts = listarCheckpoints(args.pasta, args.checkpoints, args.prefixo)
    eps = [e for e, _ in ckpts]

    secao("0. CONFIGURAÇÃO")
    comps = [len(tr) for tr in d.trajVal]
    print(f"device={device} | trajetórias de validação={len(d.trajVal)} | comprimento min/mediana/max={min(comps)}/{int(np.median(comps))}/{max(comps)}")
    print(f"DT_2C={DT_2C} SALVAR_A_CADA_2C={SALVAR_A_CADA_2C} -> DT_EFETIVO={DT_EFETIVO} | SUBPASSOS usados no alvo do treino={SUBPASSOS}")
    sdv = npy(d.escalaDelta)
    print(f"escalaDelta (espaço normalizado) min/mediana/max = {sdv.min():.3e}/{np.median(sdv):.3e}/{sdv.max():.3e}")
    print(f"passos de rollout={T} | amostras={args.amostras} | checkpoints={eps}")

    conjuntos = {
        "zero": montarConjunto(d, args.amostras, T, "zero", args.seed, device),
        "aleatorio": montarConjunto(d, args.amostras, T, "aleatorio", args.seed + 1, device),
    }
    umPasso = montarConjunto(d, args.amostras_umpasso, 1, "aleatorio", args.seed + 2, device)
    print(f"conjunto 'zero': {conjuntos['zero'].fis.shape[0]} trajetórias, início no passo 0 (as 12 primeiras são as do seu gráfico)")
    print(f"conjunto 'aleatorio': {conjuntos['aleatorio'].fis.shape[0]} janelas em posições aleatórias")

    # ---------------- 1. alvo x dataset ----------------
    secao("1. O ALVO DO TREINO (Yoshida) REPRODUZ O DATASET? (erro de 1 passo, unidades de escalaDelta)")
    ns = sorted({1, 2, 5, 10, 20, SUBPASSOS, 100, SALVAR_A_CADA_2C})
    cons = consistenciaAlvo(umPasso, d, ns)
    print(f"{'n subpassos':>12} {'rms pos':>10} {'rms vel':>10} {'|e| med pos':>12} {'|e| med vel':>12}")
    for n in ns:
        r = cons[n]
        marca = "  <- usado no treino" if n == SUBPASSOS else ("  <- SALVAR_A_CADA_2C" if n ==SALVAR_A_CADA_2C else "")
        print(f"{n:12d} {r['rms_pos']:10.4f} {r['rms_vel']:10.4f} {r['med_pos']:12.4f} {r['med_vel']:12.4f}{marca}")
    print("Leitura: o n que zera o erro é o passo interno real do gerador do dataset. Se nenhum n chega perto de 0,")
    print("o dataset e o integrador do treino diferem (epsilon, float64 vs float32, outro integrador).")

    # ---------------- 2. baselines ----------------
    secao("2. BASELINES DE ROLLOUT (sem rede), a partir do último estado real da janela")
    nsBase = sorted({2, 5, SUBPASSOS} | ({SALVAR_A_CADA_2C} if SALVAR_A_CADA_2C <= 200 else set()))
    base = {}
    for nome, c in conjuntos.items():
        for n in nsBase:
            est = rolloutIntegrador(c.ultimoFis, c.massaFis, T, n)
            base[(nome, n)] = metricasRollout(c, norm(est, d), d)
            print(resumoBase(f"[{nome}] Yoshida n={n}", base[(nome, n)]))
    print("(n=50 mostra o piso em float32; n pequeno mostra o que um integrador barato faria)")

    # perturbacao do teste de sensibilidade (mesma para a rede e para a fisica)
    ca = conjuntos["aleatorio"]
    delta = args.eps * t.randn(ca.fis.shape[0], 12, device=device) * d.escalaDelta
    sensVerd = sensibilidadeVerdadeira(ca, d, delta, T)

    # ---------------- 3. checkpoints ----------------
    secao("3. CHECKPOINTS")
    res, roll = {}, {}
    for ep, caminho in ckpts:
        ts = time.time()
        modelo = carregarModelo(caminho, device)
        c = umPasso
        with t.no_grad():
            sd = d.escalaDelta; dvR = (c.alvoNorm[:,0]-c.janela[:,-1])/sd
            ext = (c.janela[:,-1]-c.janela[:,-2])/sd
            print("extrapolar passo anterior, vel:", (ext-dvR)[:,VEL_IDX].pow(2).mean().sqrt().item())
            j2 = c.janela.clone(); j2[:,:-1] = c.janela[t.randperm(len(j2),device=j2.device),:-1]
            dif = (modelo(c.janela,c.massaNorm)-modelo(j2,c.massaNorm))/sd
            print("mudança ao trocar só o histórico, vel:", dif[:,VEL_IDX].pow(2).mean().sqrt().item())
        r = {"um": diagnosticoUmPasso(modelo, umPasso, d)}
        ruido = r["um"]["rms_comp"] * d.escalaDelta * d.desvio
        for nome, c in conjuntos.items():
            prev = rolloutLote(modelo, c.janela, c.massaNorm, T)
            roll[(ep, nome)] = prev
            r[nome] = metricasRollout(c, prev, d)
            est = rolloutIntegrador(c.ultimoFis, c.massaFis, T, SUBPASSOS, ruido)
            r[nome + "_ruido"] = metricasRollout(c, norm(est, d), d)
        r["sens"] = sensibilidadeRede(modelo, ca, d, delta, T)
        res[ep] = r
        print(f"  checkpoint {ep} ok ({time.time() - ts:.0f}s)")
        del modelo
        if device.type == "cuda":
            t.cuda.empty_cache()

    s = min(50, T)
    print(f"\nColunas: época | h_orig mediana (zero, aleat) | h erro<1%,10%,50% do tamanho (aleat) | erroPos@10, @{s} | "
          f"rms 1-passo pos, vel (unid. escalaDelta) | ganho do passo pos, vel | mediana (H-H0)/|H0| @{s} | sens rede/física @{s} | h ruído equiv.")
    for ep in eps:
        print(linhaResumo(ep, res[ep], T, sensVerd))
    best = max(eps, key=lambda e: (np.median(res[e]["aleatorio"]["h_orig"]), e))
    print(f"\nMelhor checkpoint por horizonte mediano (conjunto aleatório): época {best}")

    # ---------------- 4. detalhes ----------------
    secao(f"4. DETALHES DO MELHOR CHECKPOINT (época {best})")
    rb = res[best]
    um = rb["um"]
    print("\n-- 4.1 erro de 1 passo (entradas reais, unidades de escalaDelta) --")
    print(f"rms erro rede : pos={um['rms_pos']:.4f} vel={um['rms_vel']:.4f} | persistência: pos={um['pers_pos']:.3f} vel={um['pers_vel']:.3f}")
    print(f"viés (erro médio com sinal): pos={um['vies_pos']:+.4f} vel={um['vies_vel']:+.4f}")
    print(f"ganho do passo (inclinação previsto vs real): pos={um['ganho_pos']:.4f} vel={um['ganho_vel']:.4f} | correlação: pos={um['corr_pos']:.5f} vel={um['corr_vel']:.5f}")
    print(f"energia 1 passo: viés={um['dE_vies']:+.3e} mediana|.|={um['dE_med']:.3e} rms={um['dE_rms']:.3e} | variação por passo do próprio dataset (rms)={um['dE_dataset_rms']:.3e}")
    print("rms por componente (pos c1,c2,c3 | vel c1,c2,c3):")
    rc = npy(um["rms_comp"]).reshape(2, 6)
    for i in range(2):
        print(f"  corpo {i + 1}: pos={np.round(rc[i, :3], 4)} vel={np.round(rc[i, 3:], 4)}")

    for nome in ("aleatorio", "zero"):
        m = rb[nome]
        mr = rb[nome + "_ruido"]
        print(f"\n-- 4.2 rollout, conjunto '{nome}' --")
        imprimirCurvas("erro de posição (mediana entre amostras, unidades físicas)",
                       {"rede": med(m["erroPos"]), "ruído eq.": med(mr["erroPos"]),
                        **{f"Yoshida n={n}": med(base[(nome, n)]["erroPos"]) for n in nsBase if n != SUBPASSOS}})
        imprimirCurvas("erro de velocidade (mediana)", {"rede": med(m["erroVel"]), "ruído eq.": med(mr["erroVel"])})
        imprimirCurvas("P90 do erro de posição", {"rede": np.nanpercentile(fin(m["erroPos"]), 90, 0)})
        imprimirCurvas("(H-H0)/|H0| mediana: previsto vs dataset", {"previsto": med(m["derP"]), "dataset": med(m["derR"])})
        imprimirCurvas("|ΔP| (momento linear) e |ΔL| (momento angular) vs início, unid. de escala", {"|ΔP|": med(m["dP"]), "|ΔL|": med(m["dL"])})
        print(f"horizonte original: média={np.mean(m['h_orig']):.1f} mediana={np.median(m['h_orig']):.1f} "
              f"P10/P90={np.percentile(m['h_orig'], 10):.0f}/{np.percentile(m['h_orig'], 90):.0f} | ruído equivalente: mediana={np.median(mr['h_orig']):.1f}")
        print(f"fração de amostras em que o sistema previsto 'escapa' (distância máx > 2x real): {np.mean(m['escape']):.3f}")
        fit = ajusteCrescimento(med(m["erroPos"]))
        if fit:
            print(f"crescimento do erro de posição (passos 3..{T}): lei de potência expoente={fit['expoente']:.2f} R2={fit['r2_pot']:.3f} | "
                  f"exponencial taxa={fit['taxa']:.3f}/passo dobra={fit['dobra']:.1f} passos R2={fit['r2_exp']:.3f}")

    print("\n-- 4.3 sensibilidade a perturbações (rede vs sistema real) --")
    imprimirCurvas(f"||Δ saída||/||Δ entrada|| mediana (perturbação inicial = {args.eps} x escalaDelta)",
                   {"rede": med(rb["sens"]), "física": med(sensVerd)})

    print("\n-- 4.4 quem falha? (conjunto aleatório) --")
    f = caracteristicas(ca)
    print(f"{'característica':<16}{'spearman c/ h_orig':>22}{'spearman c/ h10%':>20}")
    for k, v in f.items():
        print(f"{k:<16}{spearman(v, rb['aleatorio']['h_orig']):22.3f}{spearman(v, rb['aleatorio']['h10']):20.3f}")
    tabelaQuartis("dist_min", f["dist_min"], rb["aleatorio"]["h_orig"])
    tabelaQuartis("vel_tipica", f["vel_tipica"], rb["aleatorio"]["h_orig"])
    tabelaQuartis("razao_massas", f["razao_massas"], rb["aleatorio"]["h_orig"])

    print("\n-- 4.5 evolução entre checkpoints: horizonte (aleatório) e erro de posição @10/@50 --")
    for ep in eps:
        a = res[ep]["aleatorio"]
        print(f"  ép {ep:3d}: h_orig mediana={np.median(a['h_orig']):5.1f} | errPos@10={med(a['erroPos'])[9]:.3e} @{s}={med(a['erroPos'])[s - 1]:.3e} "
              f"| 1-passo pos={res[ep]['um']['rms_pos']:.4f} vel={res[ep]['um']['rms_vel']:.4f}")

    # ---------------- 5. leitura automatica ----------------
    secao("5. LEITURA AUTOMÁTICA (heurísticas)")
    for linha in leituraAutomatica(res, base, cons, best, T, sensVerd):
        print("- " + linha)

    # ---------------- figuras / arquivos ----------------
    umRef = (cons[SUBPASSOS]["rms_pos"], cons[SUBPASSOS]["rms_vel"])
    figuraCurvas(res, base, eps, best, T, umRef, sensVerd, saida)
    figurasTrajetorias(res, roll, conjuntos["zero"], best, d, saida)
    figuraFalhas(ca, rb["aleatorio"], saida, best)
    flat = {}
    for ep in eps:
        for nome in ("zero", "aleatorio", "zero_ruido", "aleatorio_ruido"):
            for k, v in res[ep][nome].items():
                flat[f"ep{ep}_{nome}_{k}"] = v
        flat[f"ep{ep}_sens"] = res[ep]["sens"]
    flat["sens_verdadeira"] = sensVerd
    np.savez_compressed(saida / "resultados.npz", **flat)

    secao("FIM")
    print(f"tempo total: {time.time() - t0:.0f}s")
    print(f"arquivos em {saida}/: relatorio.txt curvas.png trajetorias.png energia_trajetorias.png falhas.png resultados.npz")


if __name__ == "__main__":
    main()