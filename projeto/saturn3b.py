#!/usr/bin/env python3
"""
saturn3.py -- rede de 3 corpos com janela 1.

Mudancas em relacao ao saturn2.py:
  1. janela = 1 estado (o sistema e markoviano); 3 tokens, um por corpo, entao a atencao troca informacao ENTRE corpos
  2. entradas: coordenadas relativas ao centro de massa, |r_ij|, 1/r_ij, 1/r_ij^3, log massas, m_i/M
  3. loss de velocidade RELATIVA por corpo (erro / |dv verdadeiro|, com piso na escala de massa) + Huber
  4. sem scheduled sampling: o alvo e sempre integrar(estado_de_entrada), entao posso perturbar o estado de
     entrada com ruido e o alvo continua exato (simula os erros do rollout sem rollout)
  5. AdamW + warmup + cosseno + EMA dos pesos; amostragem direta de estados (sem DataLoader)

Uso:   python3 saturn3.py --passos 100000
Retoma sozinho do ultimo cpasso*.pt da pasta. Checkpoints: cpasso{passo/1000:04d}k.pt
Diagnostico: python3 diagnostico.py --prefixo cpasso --checkpoints 20 40 60 80 100   (numeros = milhares de passos)
"""
import argparse
import copy
import math
import re
import time
from pathlib import Path

import numpy as np
import torch as t

import saturn2 as S 
from config import *

PASTA_DADOS = "simulacoesArtificiais/simulacoes2C"
DT_EFETIVO = S.DT_2C * S.SALVAR_A_CADA_2C
SUBPASSOS = 50
# 8 (o próprio corpo - 3 posições relativas ao centro de massa, 3 velocidades relativas, log m, m/M) + 15 ()
# 15 (o outro corpo - DR vetor até o outro corpo, DV vel relativa, d distância, 1/d, dr/d³, log d, log m_j, m_j/M, (dr*dv)/d)
FEAT = 23       #é preciso escrever à mão pois o python não vai contar sozinho os features

PESOS_LOSS = {"vel": 1.0, "pos": 1.0, "energia": 1.2, "momLin": 0.3, "momAng": 0.3}
#tolerâncias
TOL_E = 1e-4      # erro relativo de energia que conta como "1" 
TOL_PL = 1e-3     # momentos: em fração de kV (escala típica de dv)
CHAVES = list(PESOS_LOSS)

# ======================= FUNÇÕES PARA 2 CORPOS =======================
def calcularHamiltoniano(estado, massas):    
    r1=estado[...,0:3]
    v1=estado[...,3:6]
    r2=estado[...,6:9]
    v2=estado[...,9:12]
    
    m1,m2=massas[...,0],massas[...,1]
    r12=t.sqrt(t.linalg.norm(r2-r1, dim=-1)**2+epsilon**2)

    energiaCineticaSistema=(0.5*m1*t.sum(v1**2, dim=-1)+0.5*m2*t.sum(v2**2, dim=-1))                 
    energiaPotencialSistema=-G*(m1*m2/r12)
    return energiaCineticaSistema+energiaPotencialSistema
    
def calcularMomentoLinearVetor(estado, massas):      
    v1=estado[...,3:6]
    v2=estado[...,9:12]
    m1,m2=massas[...,0],massas[...,1]
    return (m1[...,None]*v1 + m2[...,None]*v2)

def calcularMomentoLinear(estado, massas):    
    return t.linalg.norm(calcularMomentoLinearVetor(estado,massas),dim=-1)

def calcularMomentoAngular(estado, massas):     
    r1=estado[...,0:3]
    v1=estado[...,3:6]
    r2=estado[...,6:9]
    v2=estado[...,9:12]
    m1,m2=massas[...,0],massas[...,1]
    return m1[...,None]*t.cross(r1,v1,dim=-1)+m2[...,None]*t.cross(r2,v2,dim=-1)

def atualizaAceleracoes_posicoes2Corpos(r1,r2,m1,m2):
    #vetor posição
    pr12=r2-r1
    #radicando
    rr12=t.sum(pr12**2,dim=-1)+epsilon**2

    #fazendo o inverso para poupar algumas divisões
    inv_r12=1/(rr12*t.sqrt(rr12))

    a1=G*(m2[...,None]*pr12*inv_r12[...,None])
    a2=G*(-m1[...,None]*pr12*inv_r12[...,None])
    
    return a1,a2

def integradorYoshida4aOrdem(estado,dt,m1,m2): 
    w1=1/(2-2**(1/3))
    w0=-2**(1/3)/(2-2**(1/3))
    c1=c4=w1/2
    c2=c3=(w0+w1)/2
    d1=d3=w1
    d2=w0
    #estado=estado.clone()

    r1=estado[...,0:3]
    v1=estado[...,3:6]
    r2=estado[...,6:9]
    v2=estado[...,9:12]

    # PRIMEIRA PARTE
    #DRIFT
    r1=r1+c1*dt*v1
    r2=r2+c1*dt*v2

    a1,a2=atualizaAceleracoes_posicoes2Corpos(r1,r2,m1,m2)
    #KICK
    v1=v1+d1*dt*a1
    v2=v2+d1*dt*a2

    # SEGUNDA PARTE
    #DRIFT
    r1=r1+c2*dt*v1
    r2=r2+c2*dt*v2

    a1,a2=atualizaAceleracoes_posicoes2Corpos(r1,r2,m1,m2)
    #KICK
    v1=v1+d2*dt*a1
    v2=v2+d2*dt*a2

    # TERCEIRA PARTE
    #DRIFT
    r1=r1+c3*dt*v1
    r2=r2+c3*dt*v2

    a1,a2=atualizaAceleracoes_posicoes2Corpos(r1,r2,m1,m2)
    #KICK
    v1=v1+d3*dt*a1
    v2=v2+d3*dt*a2

    # QUARTA PARTE
    #DRIFT
    r1=r1+c4*dt*v1
    r2=r2+c4*dt*v2

    return t.cat([r1,v1,r2,v2],dim=-1)

def _integrarEager(estado, dt, m1, m2, n):
    for _ in range(n):
        estado = integradorYoshida4aOrdem(estado, dt/n, m1, m2)
    return estado

class _IntegradorCUDAGraph:
    def __init__(self):
        self.cache = {}

    @t.no_grad()
    def __call__(self, estado, dt, m1, m2, n):
        if estado.device.type != "cuda":
            return _integrarEager(estado, dt, m1, m2, n)
        chave = (tuple(estado.shape), float(dt), n)
        if chave not in self.cache:
            # buffers estáticos: o grafo grava os endereços de memória, então toda chamada copia para eles
            sE = estado.clone()
            sm1, sm2 = m1.clone(), m2.clone()
            # aquecimento em stream separada (exigido antes da captura)
            s = t.cuda.Stream()
            s.wait_stream(t.cuda.current_stream())
            with t.cuda.stream(s):
                for _ in range(2):
                    _integrarEager(sE, dt, sm1, sm2, n)
            t.cuda.current_stream().wait_stream(s)
            g = t.cuda.CUDAGraph()
            with t.cuda.graph(g):
                sSaida = _integrarEager(sE, dt, sm1, sm2, n)
            self.cache[chave] = (g, sE, sm1, sm2, sSaida)
        g, sE, sm1, sm2, sSaida = self.cache[chave]
        sE.copy_(estado); sm1.copy_(m1); sm2.copy_(m2)
        g.replay()
        return sSaida.clone()   # clone: a saída estática é sobrescrita no próximo replay

_integradorGrafo = _IntegradorCUDAGraph()

def integrar(estado, dt, m1, m2, n):
    return _integradorGrafo(estado, dt, m1, m2, n)

def carregarTrajetorias(pastaNPZ):
    trajetorias,massasLista=[],[]
    for arquivo in sorted(Path(pastaNPZ).glob("*.npz")):    #pega um arquivo .npz aleatório para termos mais uma camada de aleatoriedade
        dados = np.load(arquivo)
        estado = dados["estado"]
        #print("FORMATO DO ESTADO",estado.shape[0])
        if estado.ndim < 2 or estado.shape[0] == 0:         #verificação de tamanho inválido --> caso o arquivo tenha a primeira coluna sendo igual a zero isso indica que a simulação foi falha ou já começou em uma colisão ou que ela ocorreu logo nos primeiros passos
            #print(f"[AVISO] pulando {arquivo.name}: trajetória vazia/colisão premtura demais para Yoshida gerar dados válidos")
            continue
        estado=estado[:,:12]                             #pega apenas os dados que importam --> posições e velocidades
        trajetorias.append(estado)
        massasLista.append(dados["massas"])
    return trajetorias,massasLista
    
# ======================= features =======================
def features(est, m):
    """est (B,12) fisico, m (B,2) fisico -> (B,2,FEAT), um token por corpo."""
    B = est.shape[0]
    x = est.reshape(B, 2, 6)
    r, v = x[..., :3], x[..., 3:]
    M = m.sum(1, keepdim=True)
    w = (m / M)[..., None]
    rc = r - (w * r).sum(1, keepdim=True)          # posicao relativa ao centro de massa
    vc = v - (w * v).sum(1, keepdim=True)
    f = [rc, vc, t.log(m)[..., None], (m / M)[..., None]]       #8 números que vão para o 
    j = (t.arange(2, device=est.device) + 1) % 2
    dr, dv = r[:, j] - r, v[:, j] - v
    d = (dr.pow(2).sum(-1, keepdim=True) + 1e-12).sqrt()
    f += [dr, dv, d, 1 / d, dr / d ** 3, t.log(d), t.log(m[:, j])[..., None], (m[:, j] / M)[..., None], (dr * dv).sum(-1, keepdim=True) / d]
    return t.cat(f, -1)     #concatena tudo em um único tensor

#aqui chamo o integrar() do saturn2, um yoshida de 50 subpassos no CUDA 
def alvoDe(estado, m):
    return integrar(estado, DT_EFETIVO, m[:, 0], m[:, 1], SUBPASSOS)

# ======================= modelo =======================
class RedeCorpos(t.nn.Module):
    def __init__(self, dModel=256, nHeads=8, nCamadas=6, dimFF=1024):
        super().__init__()
        #kv e kp são as escalas dos deltas da posição e velocidade
        #fMed e fStd normalizam as features
        for nome, forma in (("media", (12,)), ("desvio", (12,)), ("mediaMassa", (2,)), ("desvioMassa", (2,)), ("fMed", (FEAT,)), ("fStd", (FEAT,)), ("kV", ()), ("kP", ())):
            um = nome in ("desvio", "desvioMassa", "fStd")
            self.register_buffer(nome, (t.ones if um else t.zeros)(forma))
        self.entrada = t.nn.Linear(FEAT, dModel)        #joga as 23 features para 256 dimensões do Transformer
        #o Transformer usa a função de ativação GELU ao invés da RELu, com dropout=0
        camada = t.nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=dimFF, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        #foi posto o "enabe_nested_tensor=false" pois gerava um aviso uqe podia ser ignorado
        self.encoder = t.nn.TransformerEncoder(camada, nCamadas, norm=t.nn.LayerNorm(dModel), enable_nested_tensor=False)
        
        #cabeca leva cada token a 6 números, os deltaPosição e deltaVelocidade com pesos e bias zerados
        self.cabeca = t.nn.Linear(dModel, 6)
        t.nn.init.zeros_(self.cabeca.weight)
        t.nn.init.zeros_(self.cabeca.bias)


    #a rede devolve um numero Z por corpo, o qual é multiplicado por uma escala
    # para virar a mudança de velocidade deltaV em unidades físicas; cada termo 
    #da soma vem de OUTRO corpo, e com isso a rede não precisa aprender uma escala
    # para a relação massa X aceleração aplicada a cada corpo, uma vez que, caso
    # a massa do corpo i seja 0.8 (a soma de todas é 1), a aceleração aplicada a 
    #ele é equivalente à proporção de 0.2 (restante da massa dos outros 2 planetas),
    # dessa forma, quanto menor o corpo maior será a proporção da massa nos outros
    #2 planetas, aumentando a força (aceleração) que será aplicada a ela
    def escalaVel(self, m):
        """escala tipica de dv por corpo, so a partir das massas: kV * (M - m_i)"""
        return self.kV * (m.sum(1, keepdim=True) - m)

    def passoFis(self, estado, m):      #recebe o estado físico (B,12) e as mmassas físicas (B,3)
        B = estado.shape[0]             #guarda o tamanho do lote, para usar no reshape do return para entregar o estado físico
        x = (features(estado, m) - self.fMed) / self.fStd       #features subtraídas da média e divididas pelo desvio, gerando os dados que serão passados à entrada com média +-0 e desvio +-1 que é exatamente a ordem que a rede aprende bem
        z = self.cabeca(self.encoder(self.entrada(x)))                        # (B,3,6); z guarda, por corpo, 6 números sem unidade física, com os 3 primeiros virando a MUDANÇA DE POSIÇÃO e os 3 últimos a MUDANÇA DE VELOCIDADE
        dr = z[..., :3] * self.kP
        dv = z[..., 3:] * self.escalaVel(m)[..., None]
        return (estado.reshape(B, 2, 6) + t.cat([dr, dv], -1)).reshape(B, 12)       #devolve o próximo estado físico

    #pega o último estado da janela, o desnormaliza, chama passoFis e normaliza a saída
    def forward(self, janela, massa):
        """compativel com o diagnostico.py: entradas normalizadas, usa SO o ultimo estado da janela."""
        est = janela[:, -1] * self.desvio + self.media
        m = massa * self.desvioMassa + self.mediaMassa
        return (self.passoFis(est, m) - self.media) / self.desvio

# ======================= dados =======================
#divide as trajetórias do mesmo jeito do saturn2, com as primeiras len//5 sendo validação
def prepararDados(device):
    traj, mas = carregarTrajetorias(PASTA_DADOS)
    nVal = max(1, len(traj) // 5)                   # mesma divisao do saturn2/diagnostico
    trV, maV, trT, maT = traj[:nVal], mas[:nVal], traj[nVal:], mas[nVal:]
    #acahatar empilha todos os estados de todas as trajetórias em uma matriz (+- 4 milhoes x 12), comas massas repetidas em M. x[:-1] descarta o último estado de cada trajetória
    #"np.tile(m, (len(x) - 1, 1))" gera um array de m repetidos len(x)-1 vezes na vertical e 1 vez na horizontal, "m = np.array([2, 5, 8])  x = np.array([10, 20, 30, 40])  resultado = np.tile(m, (len(x) - 1, 1))  print(resultado)
    #[[2 5 8]
    #[2 5 8]
    #[2 5 8]]
    def achatar(tr, ms):
        E = np.concatenate([x[:-1] for x in tr]).astype(np.float32)
        M = np.concatenate([np.tile(m, (len(x) - 1, 1)) for x, m in zip(tr, ms)]).astype(np.float32)
        return t.as_tensor(E, device=device), t.as_tensor(M, device=device)

    media, desvio = S.calcularNormalizacao(trT)
    mm, dm = np.mean(maT, axis=0), np.std(maT, axis=0)
    dm = np.where(dm < 1e-8, 1.0, dm)
    return achatar(trT, maT), achatar(trV, maV), (media, desvio, mm, dm)

#roda uma vez antes de treinar, gerando as escalas de posição e velocidade
#quando eu for retomar o treino essa função vai realmente recalcular as estimatvas, mas elas vão ser sobrescritas pelos valores do ckpt depois, então está tudo certo
@t.no_grad()
def ajustarEscalas(modelo, E, Mm, norm, batch, nBatches=20):
    media, desvio, mm, dm = norm
    for nome, val in (("media", media), ("desvio", desvio), ("mediaMassa", mm), ("desvioMassa", dm)):
        getattr(modelo, nome).copy_(t.as_tensor(val, dtype=t.float32))
    idx = t.randint(0, len(E), (nBatches * batch,), device=E.device)
    F_, dr, dv = [], [], []
    for k in range(nBatches):
        i = idx[k * batch:(k + 1) * batch]
        e, m = E[i], Mm[i]
        d = (alvoDe(e, m) - e).reshape(-1, 2, 6)
        F_.append(features(e, m).reshape(-1, FEAT))
        dr.append(d[..., :3].reshape(-1))
        dv.append((d[..., 3:] / (m.sum(1, keepdim=True) - m)[..., None]).reshape(-1))
    f = t.cat(F_)
    modelo.fMed.copy_(f.mean(0))
    #lembre: o .clamp_min(1e-6) pega os valores que são maiores que 1e-6 apenas, se for menor ele é substituido por 1e-6
    modelo.fStd.copy_(f.std(0).clamp_min(1e-6))
    modelo.kP.copy_(t.cat(dr).std())
    modelo.kV.copy_(t.cat(dv).std())

#serve para rotacionar aleatoriamente as posições e velocidades para permitir um treino exposto à necessidade de compreensão do ISOTROPISMO da gravidade
def rotacionar(estado):
    #rotacao/reflexao aleatoria por amostra: a gravidade e isotrópica, entao o alvo continua valido
    #Só usar se o dataset for 3D (se for planar, isso vai gerar estados fora da distribuicao)
    B = estado.shape[0]
    #t.randn(B, 3, 3, device=estado.device) - o randn retorna uma matrizz 3x3 aleatória por amostra e a multiplica (.) por uma matriz ortogonal (que preserva comprimentos e produtos escalares, logo as distâncias e energia ficam iguais) - uma rotação - retornando apenas o Q, que é a matriz ortogonal em si que faz a rotação
    #EXEMPLO DE "Q":
    #Q = [[ 0, 1, 0],
    #     [-1, 0, 0],
    #     [ 0, 0, 1]]
    Q, _ = t.linalg.qr(t.randn(B, 3, 3, device=estado.device))
    x = estado.reshape(B, 2, 6)
    #x[..., :3] @ Q é multiplicação de matrizes em lote, com cada vetor linha sendo multiplicado pela Q de sua amostra, rotacionando assim a posição
    #x[..., 3:] @ Q é o mesmo mas com as velocidades
    return t.cat([x[..., :3] @ Q, x[..., 3:] @ Q], -1).reshape(B, 12)


# ======================= loss =======================
'''def lossPasso(modelo, prev, alvo, est, m, modo, piso, wPos):
    B = est.shape[0]
    p, a, e = (x.reshape(B, 2, 6) for x in (prev, alvo, est))   #previsto, alvo e estado de entrada
    tip = modelo.escalaVel(m)[..., None]                         # (B,3,1) escala tipica por massa
    if modo == "rel":
        esc = (a - e)[..., 3:].norm(dim=-1, keepdim=True).clamp_min(piso * tip).detach()    #uso detach para evitar que a escala receba gradiente
    else:                                                        # ablation: so a escala por massa
        esc = tip   
    errV = (p - a)[..., 3:] / esc       #erro da velocidade dividido pela escala
    lv = t.nn.functional.smooth_l1_loss(errV, t.zeros_like(errV), beta=1.0)     #smooth_l1_loss é quadrática para erros pequenos e linear para os grandes, reduzindo o efeito de outliers
    lp = (((p - a)[..., :3] / modelo.kP) ** 2).mean()
    return lv + wPos * lp, lv.detach(), lp.detach()     #os detach evitam que haja gradiente em lv e lp, as losses de velocidade e posição
'''
def lossPasso(modelo, prev, alvo, est, m, modo, piso, rampa=1.0):
    B = est.shape[0]
    p, a, e = (x.reshape(B, 2, 6) for x in (prev, alvo, est))
    tip = modelo.escalaVel(m)[..., None]
    if modo == "rel":
        esc = (a - e)[..., 3:].norm(dim=-1, keepdim=True).clamp_min(piso * tip).detach()
    else:
        esc = tip
    errV = (p - a)[..., 3:] / esc
    huber = t.nn.functional.smooth_l1_loss
    termos = {
        "vel": huber(errV, t.zeros_like(errV), beta=1.0),
        "pos": (((p - a)[..., :3] / modelo.kP) ** 2).mean(),
    }
    z = lambda x: t.zeros_like(x)
    if PESOS_LOSS["energia"] > 0:
        Hp, Ha = calcularHamiltoniano(prev, m), calcularHamiltoniano(alvo, m)
        eE = (Hp - Ha) / (Ha.abs().clamp_min(1e-3) * TOL_E)
        termos["energia"] = huber(eE, z(eE), beta=1.0)
    else:
        termos["energia"] = t.zeros((), device=est.device)
    tol = TOL_PL * modelo.kV
    if PESOS_LOSS["momLin"] > 0:
        eP = (calcularMomentoLinearVetor(prev, m) - calcularMomentoLinearVetor(alvo, m)) / tol
        termos["momLin"] = huber(eP, z(eP), beta=1.0)
    else:
        termos["momLin"] = t.zeros((), device=est.device)
    if PESOS_LOSS["momAng"] > 0:
        eL = (calcularMomentoAngular(prev, m) - calcularMomentoAngular(alvo, m)) / tol
        termos["momAng"] = huber(eL, z(eL), beta=1.0)
    else:
        termos["momAng"] = t.zeros((), device=est.device)
    loss = sum(PESOS_LOSS[k] * termos[k] * (1.0 if k in ("vel", "pos") else rampa) for k in CHAVES)
    return loss, {k: v.detach() for k, v in termos.items()}


#pega n=8 batches de estados, de val ou de treino - mede durante o treino o quanto o modelo erra a mudança de velocidade em um único passo, roda a cada "--val_cada" passos
@t.no_grad()
def validar(m_, Ev, Mv, batch, n=8, nome="VAL"):
    rel, rank, ab = [], [], []
    idx = t.randint(0, len(Ev), (n * batch,), device=Ev.device)
    for k in range(n):
        i = idx[k * batch:(k + 1) * batch]
        e, m = Ev[i], Mv[i]
        a, p = alvoDe(e, m), m_.passoFis(e, m)
        dvT = (a - e).reshape(-1, 2, 6)[..., 3:]
        dvP = (p - e).reshape(-1, 2, 6)[..., 3:]
        err = (dvP - dvT).norm(dim=-1)
        rel.append(err / dvT.norm(dim=-1).clamp_min(1e-9))
        ab.append(err / m_.escalaVel(m))
        rank.append(m.argsort(1).argsort(1))
    rel, rank, ab = t.cat(rel), t.cat(rank), t.cat(ab)
    por = " ".join(f"{nome}={rel[rank == k].median().item():.4f}" for k, nome in enumerate(("leve", "pesado")))
    return (f"  {nome:6s} erro relativo de dv (mediana) {por} | geral={rel.median().item():.4f} | rms(dv err / escala massa)={ab.pow(2).mean().sqrt().item():.4f} | mediana={ab.median().item():.4f}")


# ======================= treino =======================
#o que o main está fazendo, apenas para compreender o que tem aqui:
#lê os argumentos, carrega os dados, cria o modelo, o EMA, o otimizador e 
#o agendador de LR, retoma de um checkpoint caso haja um, e repete até o fim os
# passos de sortear estados, perturbá-los, calcular o alvo exato, prever, calcular
# a loss e atualiza os pesos
def main():
    #o argparse lê as flags e passa elas para o dModel
    ap = argparse.ArgumentParser()
    ap.add_argument("--passos", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ruido", type=float, default=0.5, help="ruido maximo no estado de entrada, em escalas de dv/dr por passo")
    ap.add_argument("--piso", type=float, default=0.05, help="piso da escala relativa, fracao da escala tipica de massa")
    #ap.add_argument("--wpos", type=float, default=1.0)
    ap.add_argument("--loss", choices=["rel", "abs"], default="rel")
    ap.add_argument("--rot", action="store_true", help="aumento por rotacao (so se o dataset for 3D)")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val_cada", type=int, default=2000)
    ap.add_argument("--salvar_cada", type=int, default=5000)
    ap.add_argument("--dmodel", type=int, default=256)
    ap.add_argument("--camadas", type=int, default=6)
    ap.add_argument("--ff", type=int, default=1024)
    a = ap.parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    #o prepararDados retorna os estados e massas de TREINO, de VALIDAÇÃO e os dados de NORMALIZAÇÃO
    (E, Mm), (Ev, Mv), norm = prepararDados(device)
    print(f"estados de treino={len(E)} | validacao={len(Ev)} | device={device}")

    configuracao = dict(dModel=a.dmodel, nHeads=8, nCamadas=a.camadas, dimFF=a.ff)  #dicionário de hiperparametros da rede já com o nheads fixo em 8 - vai ser salvo no ckpt para que o diagnostico e o plot recriem a rede
    modelo = RedeCorpos(**configuracao).to(device)      #cria a rede com valores zeerados e joga em modelo
    ajustarEscalas(modelo, E, Mm, norm, a.batch)
    print(f"kP={modelo.kP.item():.4e}  kV={modelo.kV.item():.4e}  params={sum(p.numel() for p in modelo.parameters()) / 1e6:.2f}M")
    #EMA -média móvel exponencial- é uma cópia profunda do modelo, já em eval() 
    #e sem gradiente, guarda os pesos ao longo do treino. É uma segunda cópia da
    # rede com pesos sendo uma média suavizada dos pesos recentes do modelo, ela
    # não é treinada, só acompanha o modelo, com o ruído passo a passo se
    # cancelando na média. É ela que vai para os ckpt, validação, diagnóstico e plot,
    # e não o modelo
    ema = copy.deepcopy(modelo).eval()      #cria uma cópia independente com os buffers fMed, dV e outros já ajustados 
    for p in ema.parameters():
        p.requires_grad_(False)     #impede que o EMA entre no cálculo do gradiente 
    otimizador = t.optim.AdamW(modelo.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=1e-4)
    #fLR é o fator que multiplica o LR no passo s
    #AQUECIMENTO: min(1.0, (s + 1) / a.warmup) sobe linearmente de 0 a 1 nos primeiros 1000 passos
    #COSSENO: cai de 1 a 0.03 de acordo com a função cosseno relativa ao passo s
    fLR = lambda s: min(1.0, (s + 1) / a.warmup) * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.passos))))
    #LambdaLR(otimizador, fLR) aplica esse fator a cada scheduler.step()
    scheduler = t.optim.lr_scheduler.LambdaLR(otimizador, fLR)

    passo = 0
    #retoma os ckpts na ordem crescente, pegando o último que aparece 
    ckpts = sorted(Path(".").glob("cpasso*.pt"))
    if ckpts:
        ckpt = t.load(ckpts[-1], map_location=device, weights_only=False)
        modelo.load_state_dict(ckpt["bruto"]); ema.load_state_dict(ckpt["modelo"]); otimizador.load_state_dict(ckpt["opt"])
        passo = ckpt["passo"]
        for _ in range(passo):
            scheduler.step()
        print(f"retomando de {ckpts[-1].name} (passo {passo})")

    B = a.batch     #tamanho do lote
    emaP, modP = list(ema.parameters()), list(modelo.parameters())
    #t0 é o relógio para contar as iterações por segundo para colocar no relatório
    #acc é o acumulador GPU de 3 valores - loss, lv, lp
    t0, acc = time.time(), t.zeros(1+len(CHAVES), device=device) 
    while passo < a.passos:
        modelo.train()
        #pega 4096 índices com reposição da tabela E
        i = t.randint(0, len(E), (B,), device=device)
        estado, m = E[i], Mm[i]
        #aqui ocorre a perturbação dos estados
        with t.no_grad():
            #giro do estado
            if a.rot:
                estado = rotacionar(estado)
            #tip=escalaVel(m) é a escala de deltaV por corpo
            tip = modelo.escalaVel(m)[..., None].expand(B, 2, 3)
            #sig junta a escala de posição (kP) nas 3 primeiras colunas e a de velocidade nas 3 últimas
            sig = t.cat([modelo.kP.expand(B, 2, 3), tip], -1)
            
            u = (t.rand(B, 1, 1, device=device) ** 2) * a.ruido * (t.rand(B, 1, 1, device=device) > 0.3)  # 30% limpo pois o rand > 0.3 implica em pelo menos 30% das amostras sem ruído
            #estado + u*sig*randn soma o ruído gaussiano
            estado = (estado.reshape(B, 2, 6) + u * sig * t.randn(B, 2, 6, device=device)).reshape(B, 12)
            #cálculo do próximo estado exato a partir do estado já perturbado
            alvo = alvoDe(estado, m)
        prev = modelo.passoFis(estado, m)
        #loss, lv, lp = lossPasso(modelo, prev, alvo, estado, m, a.loss, a.piso, a.wpos)
        rampa=min(1.0,max(0.0,(passo-3000)/5000))
        loss, termos = lossPasso(modelo, prev, alvo, estado, m, a.loss, a.piso, rampa)
        
        #aqui é a atualização 
        otimizador.zero_grad(set_to_none=True)
        loss.backward()
        t.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otimizador.step()   #atualiza os pesos
        scheduler.step()    #avança o LR
        passo += 1
        with t.no_grad():
            #emaP e modP são listas de parâmetros já criadas
            d = min(a.ema, (1 + passo) / (10 + passo))
            #_foreach_lerp_ aplica ema ← ema + (1 − d)(modelo − ema) a todos os tensores de uma vez,
            # em poucas chamadas de GPU, em vez de um laço Python
            t._foreach_lerp_(emaP, modP, 1 - d)
            #acc += t.stack([loss.detach(), lv, lp])
            acc += t.stack([loss.detach()] + [termos[k] for k in CHAVES])
        if passo % 200 == 0:
            v = (acc / 200).tolist(); acc.zero_()
            det = " ".join(f"{k} {x:.2e}" for k, x in zip(CHAVES, v[1:]))
            print(f"passo {passo:7d} | loss {v[0]:.3e} ({det}) | lr {scheduler.get_last_lr()[0]:.2e} "
                f"| {passo / (time.time() - t0 + 1e-9):.1f} it/s", flush=True)        
        if passo % a.val_cada == 0:
            print(validar(ema, Ev, Mv, B), flush=True)
            print(validar(ema, E, Mm, B, nome="TREINO"), flush=True)
        if passo % a.salvar_cada == 0 or passo == a.passos:
            nome = f"cpasso{passo // 1000:04d}k.pt"
            t.save(dict(modelo=ema.state_dict(), bruto=modelo.state_dict(), opt=otimizador.state_dict(), passo=passo, config=configuracao), nome)
            print("salvo", nome, flush=True)


if __name__ == "__main__":
    main()