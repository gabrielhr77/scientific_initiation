#!/usr/bin/env python3
"""
plotar_trajetoria.py -- trajetoria REAL (dataset) x PREVISTA pela rede, em 3D.

A rede recebe so o estado inicial e prevê o proximo estado, 100 vezes seguidas (cada saida vira a proxima entrada).

Exemplos (na pasta do projeto, onde estao saturn2.py, saturn3.py e os cpasso*.pt):
    python3 plotar_trajetoria.py                                  # ultimo cpasso*.pt, validacao 0,1,2, 100 passos
    python3 plotar_trajetoria.py --ckpt cpasso0030k.pt --traj 0 5 9
    python3 plotar_trajetoria.py --aleatorias 12                  # 12 trajetorias de validacao sorteadas (sem escolher a dedo)
    python3 plotar_trajetoria.py --arquivo simulacoesArtificiais/simulacoes3C/simulacao3D_123.npz
    python3 plotar_trajetoria.py --mostrar                        # abre janela 3D giravel (precisa de display)

Saidas em plots_traj/: traj_<nome>.png (3D + erro + energia) e grade.png (varias trajetorias lado a lado).
Linha cheia = real; tracejada = rede; bolinha = real, x = rede, a cada --marcar passos (mostra a defasagem de fase).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
if "--mostrar" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch as t

#import saturn2 as S
from versoes_treino_redes import saturn2 as S
import saturn3 as S3

PASTA_DADOS = "simulacoesArtificiais/simulacoes3C"
POS = [0, 1, 2, 6, 7, 8, 12, 13, 14]
CORES = ["C0", "C3", "C2"]
DT_EFETIVO = S.DT_3C * S.SALVAR_A_CADA_3C
warnings.filterwarnings("ignore", message="Tight layout")


# ======================= modelo e dados =======================
def carregarModelo(caminho, dev):
    ck = t.load(caminho, map_location=dev, weights_only=False)
    modelo = S3.RedeCorpos(**ck["config"]).to(dev)
    modelo.load_state_dict(ck["modelo"])
    modelo.eval()
    return modelo, ck.get("passo")


def escolherTrajetorias(a):
    """lista de (nome, estados (T,18), massas (3,))"""
    if a.arquivo:
        out = []
        for caminho in a.arquivo:
            d = np.load(caminho)
            out.append((Path(caminho).stem, d["estado"][:, :18].astype(np.float32), np.asarray(d["massas"], np.float32)))
        return out
    trajs, mas = S.carregarTrajetorias(PASTA_DADOS)
    nVal = max(1, len(trajs) // 5)                       # mesma divisao do treino: as primeiras len//5 sao validacao
    if a.aleatorias:
        ids = sorted(np.random.default_rng(a.seed).choice(nVal, min(a.aleatorias, nVal), replace=False).tolist())
    else:
        ids = a.traj
    bons = [i for i in ids if 0 <= i < nVal]
    if len(bons) < len(ids):
        print(f"[aviso] indices fora da validacao (0..{nVal - 1}) ignorados: {[i for i in ids if i not in bons]}")
    return [(f"val{i}", trajs[i].astype(np.float32), np.asarray(mas[i], np.float32)) for i in bons]


@t.no_grad()
def rollout(modelo, est0, massa, n, dev):
    e = t.as_tensor(est0, device=dev)[None]
    m = t.as_tensor(massa, device=dev)[None]
    out = []
    for _ in range(n):
        e = modelo.passoFis(e, m)                        # estado fisico -> proximo estado fisico
        out.append(e[0].cpu().numpy())
    return np.array(out)


# ======================= metricas =======================
def distancias(x):
    r = x[:, POS].reshape(-1, 3, 3)
    return np.stack([np.linalg.norm(r[:, 1] - r[:, 0], axis=-1),
                     np.linalg.norm(r[:, 2] - r[:, 0], axis=-1),
                     np.linalg.norm(r[:, 2] - r[:, 1], axis=-1)], -1)


def energia(x, massa):
    xt = t.as_tensor(x, dtype=t.float64)
    mt = t.as_tensor(massa, dtype=t.float64)[None].expand(len(x), 3)
    return S.calcularHamiltoniano(xt, mt).numpy()


def metricas(real, prev, massa):
    """real, prev: (n+1, 18), incluindo o estado inicial na linha 0."""
    err = np.linalg.norm((prev - real)[:, POS], axis=1)[1:]
    tam = float(distancias(real).mean())                 # tamanho tipico do sistema neste trecho
    def horizonte(frac):
        idx = np.where(err > frac * tam)[0]
        return int(idx[0]) if len(idx) else len(err)     # passos sobrevividos antes de passar do limiar
    H0 = energia(real[:1], massa)[0]
    return dict(err=err, tam=tam, h1=horizonte(0.01), h10=horizonte(0.10), h50=horizonte(0.50),
                dEreal=(energia(real, massa) - H0) / abs(H0), dEprev=(energia(prev, massa) - H0) / abs(H0))


# ======================= figuras =======================
def desenhar3D(ax, real, prev, massa, marcar=20, legenda=True, rotulos=True):
    for b in range(3):
        sl = slice(6 * b, 6 * b + 3)
        R, P, c = real[:, sl], prev[:, sl], CORES[b]
        ax.plot(*R.T, color=c, lw=2.4, alpha=.9, label=f"real, corpo {b + 1} (m={massa[b]:.3f})")
        ax.plot(*P.T, color=c, lw=1.7, ls="--", label=f"rede, corpo {b + 1}")
        idx = np.arange(marcar, len(R), marcar)
        if len(idx):
            ax.scatter(*R[idx].T, color=c, s=24, marker="o", edgecolors="k", linewidths=.6)
            ax.scatter(*P[idx].T, color=c, s=34, marker="x")
        ax.scatter(*R[0], color=c, s=90, marker="o", edgecolors="k", linewidths=1.2, zorder=6)   # inicio
        ax.scatter(*R[-1], color=c, s=60, marker="s", edgecolors="k", linewidths=.8, zorder=6)    # fim real
        ax.scatter(*P[-1], color=c, s=60, marker="^", edgecolors="k", linewidths=.8, zorder=6)    # fim rede
    pts = real[:, POS].reshape(-1, 3)                    # limites pelo trecho REAL (a rede pode fugir; ela sai do quadro)
    c0, meio = (pts.max(0) + pts.min(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2 * 1.15
    ax.set_xlim(c0[0] - meio, c0[0] + meio); ax.set_ylim(c0[1] - meio, c0[1] + meio); ax.set_zlim(c0[2] - meio, c0[2] + meio)
    ax.set_box_aspect((1, 1, 1))
    if rotulos:
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    else:
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    if legenda:
        ax.legend(fontsize=8, loc="upper left")


def figuraTrajetoria(nome, real, prev, massa, m, ckpt, marcar, saida):
    n = len(m["err"])
    fig = plt.figure(figsize=(19, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    desenhar3D(ax3, real, prev, massa, marcar)
    ax3.set_title(f"{nome} | marcas a cada {marcar} passos | circulo=inicio, quadrado=fim real, triangulo=fim rede")
    axE = fig.add_subplot(gs[0, 1])
    x = np.arange(1, n + 1)
    axE.semilogy(x, np.maximum(m["err"], 1e-12), "C3", lw=2, label="erro de posicao (norma, 3 corpos)")
    for frac, cor in ((0.01, "gray"), (0.10, "k")):
        axE.axhline(frac * m["tam"], color=cor, ls=":", lw=1, label=f"{frac:.0%} do tamanho do sistema")
    axE.axvline(m["h10"], color="k", ls="-.", lw=1)
    axE.set_title(f"erro vs passo | horizonte (erro<10%): {m['h10']} passos | (erro<1%): {m['h1']} passos")
    axE.set_xlabel("passo"); axE.legend(fontsize=8); axE.grid(True, alpha=.3)
    axH = fig.add_subplot(gs[1, 1])
    axH.plot(m["dEreal"], "k--", label="real (dataset)")
    axH.plot(np.clip(m["dEprev"], -1, 1), "C3", label="rede")
    axH.set_title("energia: (H - H0) / |H0|"); axH.set_xlabel("passo"); axH.legend(); axH.grid(True, alpha=.3)
    plt.suptitle(f"{Path(ckpt).name}: a rede preve o proximo estado {n} vezes seguidas a partir do estado inicial", y=0.995)
    plt.tight_layout()
    plt.savefig(saida / f"traj_{nome}.png", dpi=110)
    return fig


def figuraGrade(itens, marcar, saida):
    k = len(itens)
    cols = 4 if k > 6 else min(3, k)
    lin = int(np.ceil(k / cols))
    fig = plt.figure(figsize=(5.2 * cols, 5.0 * lin))
    for i, (nome, real, prev, massa, m) in enumerate(itens):
        ax = fig.add_subplot(lin, cols, i + 1, projection="3d")
        desenhar3D(ax, real, prev, massa, marcar, legenda=False, rotulos=False)
        ax.set_title(f"{nome} | h10%={m['h10']} h1%={m['h1']}", fontsize=10)
    plt.suptitle("linha cheia = real, tracejada = rede (cores = corpos); x = rede / bolinha = real a cada marca", y=0.995)
    plt.tight_layout()
    plt.savefig(saida / "grade.png", dpi=100)
    return fig


# ======================= main =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="cpasso*.pt (padrao: o ultimo da pasta)")
    ap.add_argument("--traj", type=int, nargs="*", default=[0, 1, 2], help="indices das trajetorias de VALIDACAO")
    ap.add_argument("--aleatorias", type=int, default=0, help="sorteia K trajetorias de validacao")
    ap.add_argument("--arquivo", nargs="*", default=None, help="um ou mais .npz (atencao: pode ser trajetoria de treino)")
    ap.add_argument("--passos", type=int, default=100)
    ap.add_argument("--inicio", type=int, default=0, help="indice do estado inicial dentro da trajetoria")
    ap.add_argument("--marcar", type=int, default=20)
    ap.add_argument("--saida", default="plots_traj")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mostrar", action="store_true", help="abre janelas 3D interativas (giraveis)")
    a = ap.parse_args()

    dev = t.device("cuda" if t.cuda.is_available() else "cpu")
    caminho = a.ckpt or str(sorted(Path(".").glob("cpasso*.pt"))[-1])
    modelo, passo = carregarModelo(caminho, dev)
    saida = Path(a.saida)
    saida.mkdir(exist_ok=True)
    print(f"checkpoint: {caminho} (passo {passo}) | {a.passos} passos = {a.passos * DT_EFETIVO:.2f} u.t. "
          f"= {a.passos * DT_EFETIVO / 2.828:.2f} tempos de travessia")

    itens, figs = [], []
    for nome, traj, massa in escolherTrajetorias(a):
        n = min(a.passos, len(traj) - 1 - a.inicio)
        if n < 5:
            print(f"[aviso] {nome}: trajetoria curta demais a partir de {a.inicio}, pulando")
            continue
        real = traj[a.inicio:a.inicio + n + 1]
        prev = np.vstack([real[:1], rollout(modelo, real[0], massa, n, dev)])
        m = metricas(real, prev, massa)
        rel = lambda k: m["err"][min(k, n) - 1] / m["tam"]
        print(f"{nome:>10} | m=({massa[0]:.3f},{massa[1]:.3f},{massa[2]:.3f}) | h(1%)={m['h1']:3d} h(10%)={m['h10']:3d} h(50%)={m['h50']:3d} "
              f"| erro/tamanho @25={rel(25):.3f} @50={rel(50):.3f} @75={rel(75):.3f} @{n}={rel(n):.3f}")
        itens.append((nome, real, prev, massa, m))
        fig = figuraTrajetoria(nome, real, prev, massa, m, caminho, a.marcar, saida)
        if a.mostrar:
            figs.append(fig)
        else:
            plt.close(fig)

    if not itens:
        return
    h10 = np.array([i[4]["h10"] for i in itens])
    h1 = np.array([i[4]["h1"] for i in itens])
    print(f"\nresumo ({len(itens)} trajetorias): h(10%) mediana={np.median(h10):.0f} | chegam a {a.passos} passos: {(h10 >= a.passos).mean():.0%} "
          f"| h(1%) mediana={np.median(h1):.0f}")
    if len(itens) > 1:
        fg = figuraGrade(itens, a.marcar, saida)
        if not a.mostrar:
            plt.close(fg)
    print(f"figuras em {saida}/")
    if a.mostrar:
        plt.show()


if __name__ == "__main__":
    main()