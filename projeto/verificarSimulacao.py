"""
verificar_simulacao.py
=======================
Ferramenta de verificação/validação para os arquivos .npz gerados por
geradorDataset.py (simulações de 2 e 3 corpos).

O QUE ELE FAZ
-------------
Para um arquivo de simulação (.npz), o script:
  1. Recalcula energia, momento linear e momento angular DIRETAMENTE a partir
     da trajetória bruta salva (estado), e compara com os valores que o
     próprio gerador já havia salvo -> detecta bugs de "bookkeeping".
  2. Verifica as leis de conservação (energia, momento linear, momento
     angular) ao longo de toda a simulação.
  3. Verifica se o centro de massa realmente permanece parado na origem
     (ele DEVE, pois o gerador zera posição e velocidade do CM no estado inicial
     e não há forças externas).
  4. Verifica a consistência do passo de tempo salvo, presença de NaN/Inf,
     soma das massas, e a distância mínima entre os corpos (para comparar
     com o critério de colisão e o motivoTermino).
  5. (Se rodado na pasta do projeto, com fisica2corpos.py/fisica3corpos.py
     acessíveis) faz um teste de REVERSIBILIDADE TEMPORAL: integra o estado
     inicial N passos para frente com o integrador de Yoshida e depois N
     passos para trás (dt -> -dt). Um integrador simplético/reversível
     correto deve retornar bem perto do estado original. Esse é o teste mais
     forte de que a implementação do integrador está correta.
  6. Plota a trajetória (3D e projeção XY), a distância entre os pares de
     corpos, o erro relativo de energia, o módulo do momento linear e o
     desvio relativo do momento angular, tudo ao longo do tempo.

USO
---
  # verifica o arquivo mais recente encontrado em simulacoesArtificiais/
  python verificarSimulacao.py

  # verifica um arquivo específico
  python verificarSimulacao.py simulacoesArtificiais/simulacoes3C/simulacao3D_7.npz

  # verifica TODOS os arquivos de uma pasta (modo resumido, tabela) e ao final
  # detalha automaticamente o pior caso (maior desvio de energia)
  python verificarSimulacao.py --pasta simulacoesArtificiais/simulacoes3C

  # roda de outro diretório, apontando onde estão config.py / fisica*corpos.py
  python verificarSimulacao.py --projeto-dir /caminho/do/projeto arquivo.npz

Rode este script na mesma pasta dos demais .py do projeto (config.py,
fisica2corpos.py, fisica3corpos.py) para habilitar o teste de reversibilidade
e usar exatamente o G/epsilon configurados. Sem isso, o script ainda funciona
(usa G=1, epsilon=1e-6, que são os valores padrão do config.py atual), só que
sem o teste de reversibilidade.
"""

import os
import sys
import glob
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necessário para projection='3d')


# ============================================================================
#                       CARREGAMENTO DO "PROJETO" (opcional)
# ============================================================================

def carregar_projeto(caminho_dir):
    """Tenta importar config.py, fisica2corpos.py e fisica3corpos.py da pasta
    do projeto, para usar G/epsilon reais e o integrador de Yoshida real no
    teste de reversibilidade. Se não conseguir, cai em valores padrão e
    desabilita apenas o teste de reversibilidade (o resto do script continua
    funcionando normalmente com fórmulas genéricas equivalentes)."""
    caminho_dir = os.path.abspath(caminho_dir)
    if caminho_dir not in sys.path:
        sys.path.insert(0, caminho_dir)

    projeto = {"G": 1.0, "epsilon": 1e-6, "SALVAR_A_CADA": None, "f2": None, "f3": None}

    try:
        import config as _config
        projeto["G"] = _config.G
        projeto["epsilon"] = _config.epsilon
        projeto["SALVAR_A_CADA"] = getattr(_config, "SALVAR_A_CADA", None)
        projeto["MARGEM_SEGURANCA"] = getattr(_config, "MARGEM_SEGURANCA", None)
    except Exception as e:
        print(f"[aviso] não consegui importar config.py em '{caminho_dir}' ({e}). "
              f"Usando G=1, epsilon=1e-6 (padrão do projeto).")

    try:
        import fisica2corpos as _f2
        projeto["f2"] = _f2
    except Exception as e:
        print(f"[aviso] não consegui importar fisica2corpos.py ({e}). "
              f"Teste de reversibilidade para 2 corpos será pulado.")

    try:
        import fisica3corpos as _f3
        projeto["f3"] = _f3
    except Exception as e:
        print(f"[aviso] não consegui importar fisica3corpos.py ({e}). "
              f"Teste de reversibilidade para 3 corpos será pulado.")

    return projeto


# ============================================================================
#                    FÍSICA GENÉRICA (N corpos, p/ conferência)
# ============================================================================
# Reimplementação genérica (válida para qualquer N) das mesmas fórmulas de
# fisica2corpos.py / fisica3corpos.py. Serve para RECALCULAR energia/momentos
# a partir da trajetória bruta e comparar com o que o gerador salvou, além de
# funcionar mesmo se o script rodar fora da pasta do projeto.

def extrair_corpos(estado_linha, n_corpos):
    """estado_linha: vetor 1D com 6*n_corpos elementos (sem a coluna de tempo).
    Retorna listas de posições e velocidades, uma por corpo."""
    posicoes = [estado_linha[6 * i: 6 * i + 3] for i in range(n_corpos)]
    velocidades = [estado_linha[6 * i + 3: 6 * i + 6] for i in range(n_corpos)]
    return posicoes, velocidades


def energia_generica(estado_linha, massas, G, epsilon):
    n = len(massas)
    pos, vel = extrair_corpos(estado_linha, n)
    cinetica = sum(massas[i] * np.dot(vel[i], vel[i]) / 2.0 for i in range(n))
    potencial = 0.0
    for i, j in itertools.combinations(range(n), 2):
        r = np.sqrt(np.sum((pos[j] - pos[i]) ** 2) + epsilon ** 2)
        potencial -= G * massas[i] * massas[j] / r
    return cinetica + potencial


def mom_lin_generico(estado_linha, massas):
    n = len(massas)
    _, vel = extrair_corpos(estado_linha, n)
    p = sum(massas[i] * vel[i] for i in range(n))
    return np.linalg.norm(p)


def mom_ang_generico(estado_linha, massas):
    n = len(massas)
    pos, vel = extrair_corpos(estado_linha, n)
    return sum(massas[i] * np.cross(pos[i], vel[i]) for i in range(n))


def centro_de_massa(estado_linha, massas):
    n = len(massas)
    pos, _ = extrair_corpos(estado_linha, n)
    return sum(massas[i] * pos[i] for i in range(n)) / np.sum(massas)


MOTIVO_TERMINO_TXT = {
    0: "simulação completa (rodou todos os steps sem intercorrências)",
    1: "encerrada por aproximação/colisão (trajetória cortada antes da margem de segurança)",
    2: "encerrada por energia positiva (sistema hiperbólico/não ligado)",
}


# ============================================================================
#                              CARREGAR ARQUIVO
# ============================================================================

def carregar_npz(caminho):
    dados = np.load(caminho)
    massas = np.array(dados["massas"], dtype=float)
    estado = np.array(dados["estado"], dtype=float)   # (n_pontos, 6*N + 1) -> última coluna é tempo
    tempo = np.array(dados["tempo"], dtype=float)
    energia = np.array(dados["energia"], dtype=float)
    momAng = np.array(dados["momAng"], dtype=float)
    momLin = np.array(dados["momLin"], dtype=float)
    dt = float(dados["dt"])
    motivoTermino = int(dados["motivoTermino"])
    return {
        "massas": massas, "estado": estado, "tempo": tempo, "energia": energia,
        "momAng": momAng, "momLin": momLin, "dt": dt, "motivoTermino": motivoTermino,
    }


# ============================================================================
#                                VERIFICAÇÃO
# ============================================================================

def verificar_arquivo(caminho, projeto, n_passos_reversao=3000, gerar_grafico=True,
                       salvar_fig=None, mostrar_grafico=True):
    nome_arquivo = os.path.basename(caminho)
    d = carregar_npz(caminho)
    massas, estado, tempo = d["massas"], d["estado"], d["tempo"]
    energia_salva, momAng_salvo, momLin_salvo = d["energia"], d["momAng"], d["momLin"]
    dt, motivoTermino = d["dt"], d["motivoTermino"]

    n_corpos = len(massas)
    n_pontos = estado.shape[0]
    G, epsilon = projeto["G"], projeto["epsilon"]

    # sanidade básica -------------------------------------------------------
    tem_nan = bool(np.isnan(estado).any() or np.isnan(energia_salva).any())
    tem_inf = bool(np.isinf(estado).any() or np.isinf(energia_salva).any())
    soma_massas = float(np.sum(massas))

    # passo de tempo efetivo entre pontos salvos -----------------------------
    diffs_tempo = np.diff(tempo)
    dt_efetivo_medio = float(np.mean(diffs_tempo)) if len(diffs_tempo) else float("nan")
    dt_efetivo_desvio = float(np.std(diffs_tempo)) if len(diffs_tempo) else float("nan")
    salvar_a_cada_inferido = round(dt_efetivo_medio / dt) if dt_efetivo_medio == dt_efetivo_medio else None

    # recomputa energia/momentos a partir da trajetória bruta (sem tempo) ---
    estado_sem_tempo = estado[:, :-1]
    energia_recalc = np.array([energia_generica(s, massas, G, epsilon) for s in estado_sem_tempo])
    momLin_recalc = np.array([mom_lin_generico(s, massas) for s in estado_sem_tempo])
    momAng_recalc = np.array([mom_ang_generico(s, massas) for s in estado_sem_tempo])

    erro_bookkeeping_energia = float(np.max(np.abs(energia_recalc - energia_salva)))
    erro_bookkeeping_momLin = float(np.max(np.abs(momLin_recalc - momLin_salvo)))
    erro_bookkeeping_momAng = float(np.max(np.linalg.norm(momAng_recalc - momAng_salvo, axis=-1)))

    # conservação de energia --------------------------------------------------
    E = energia_salva
    E0 = E[0]
    escala_E = max(abs(np.mean(E)), 1e-12)
    desvio_energia_rel = float((E.max() - E.min()) / escala_E)
    erro_energia_final_rel = float((E[-1] - E0) / max(abs(E0), 1e-12))
    erro_relativo_energia_t = (E - E0) / max(abs(E0), 1e-12)

    # momento linear (deveria ficar ~0, pois foi zerado no referencial do CM)
    pos0, vel0 = extrair_corpos(estado_sem_tempo[0], n_corpos)
    escala_P = max(max(massas[i] * np.linalg.norm(vel0[i]) for i in range(n_corpos)), 1e-12)
    momLin_max_abs = float(np.max(momLin_salvo))
    momLin_max_rel = float(momLin_max_abs / escala_P)

    # momento angular (deveria ser constante) --------------------------------
    L0 = momAng_salvo[0]
    escala_L = max(np.linalg.norm(L0), 1e-12)
    desvio_momAng_rel = float(np.max(np.linalg.norm(momAng_salvo - L0, axis=-1)) / escala_L)

    # deriva do centro de massa (posição) -------------------------------------
    R_cm_t = np.array([centro_de_massa(s, massas) for s in estado_sem_tempo])
    deriva_cm_max = float(np.max(np.linalg.norm(R_cm_t, axis=-1)))

    # distâncias mínimas entre pares de corpos --------------------------------
    pares = list(itertools.combinations(range(n_corpos), 2))
    distancias = {}
    for (i, j) in pares:
        pos_i = estado_sem_tempo[:, 6 * i: 6 * i + 3]
        pos_j = estado_sem_tempo[:, 6 * j: 6 * j + 3]
        distancias[(i, j)] = np.linalg.norm(pos_j - pos_i, axis=-1)
    distancia_minima_global = float(min(d_ij.min() for d_ij in distancias.values()))

    # teste de reversibilidade temporal (opcional, exige integrador real) ----
    erro_reversao = None
    if n_corpos == 2 and projeto.get("f2") is not None:
        integrador = projeto["f2"].yoshida4ordem2Corpos
    elif n_corpos == 3 and projeto.get("f3") is not None:
        integrador = projeto["f3"].yoshida4ordem3Corpos
    else:
        integrador = None

    if integrador is not None:
        k = min(n_passos_reversao, 20000)
        s = estado_sem_tempo[0].copy()
        if n_corpos == 3:
            args = (massas[0], massas[1], massas[2])
        else:
            args = (massas[0], massas[1])
        for _ in range(k):
            s = integrador(s, dt, *args)
        for _ in range(k):
            s = integrador(s, -dt, *args)
        erro_reversao = float(np.linalg.norm(s - estado_sem_tempo[0]) /
                               max(np.linalg.norm(estado_sem_tempo[0]), 1e-12))

    # ------------------------------------------------------------------ print
    print("=" * 78)
    print(f"ARQUIVO: {nome_arquivo}")
    print("=" * 78)
    print(f"Corpos simulados ......................... {n_corpos}")
    print(f"Massas .................................... {np.round(massas, 6).tolist()}  "
          f"(soma = {soma_massas:.10f}, esperado = 1.0)")
    print(f"Pontos salvos na trajetória ............... {n_pontos}")
    print(f"dt (integração) ............................ {dt:.6g}")
    print(f"Intervalo médio entre pontos salvos ....... {dt_efetivo_medio:.6g} "
          f"(desvio-padrão: {dt_efetivo_desvio:.2e})")
    if projeto.get("SALVAR_A_CADA") is not None:
        esperado = dt * projeto["SALVAR_A_CADA"]
        ok = "OK" if abs(dt_efetivo_medio - esperado) < 1e-9 else "DIVERGENTE!"
        print(f"    -> esperado (dt * SALVAR_A_CADA={projeto['SALVAR_A_CADA']}) = {esperado:.6g}  [{ok}]")
    elif salvar_a_cada_inferido is not None:
        print(f"    -> SALVAR_A_CADA inferido a partir dos dados: {salvar_a_cada_inferido}")
    print(f"Tempo total coberto ........................ {tempo[-1] - tempo[0]:.4f}")
    print(f"Motivo de término (motivoTermino={motivoTermino}) -> "
          f"{MOTIVO_TERMINO_TXT.get(motivoTermino, 'valor desconhecido')}")
    print(f"NaN/Inf na trajetória ou energia? ......... {'SIM  <-- PROBLEMA' if (tem_nan or tem_inf) else 'não'}")
    print("-" * 78)
    print("CONFERÊNCIA DO BOOKKEEPING (energia/momentos recalculados vs. salvos)")
    print(f"  máx |E_recalc - E_salva| ................ {erro_bookkeeping_energia:.3e}")
    print(f"  máx |P_recalc - P_salvo| ................ {erro_bookkeeping_momLin:.3e}")
    print(f"  máx |L_recalc - L_salvo| ................ {erro_bookkeeping_momAng:.3e}")
    print("  (esses três valores devem ser ~1e-10 ou menor; se estiverem grandes,")
    print("   há uma inconsistência entre o que é calculado e o que é salvo no gerador)")
    print("-" * 78)
    print("LEIS DE CONSERVAÇÃO")
    print(f"  Energia:")
    print(f"    E(0) = {E0:.6e}   E(fim) = {E[-1]:.6e}   {'(ligado, E<0)' if E0 < 0 else '(não ligado, E>=0)'}")
    print(f"    desvio relativo (Emax-Emin)/|Emed| ..... {desvio_energia_rel:.3e}")
    print(f"    erro relativo final (E_f - E_0)/|E_0| .. {erro_energia_final_rel:.3e}")
    print(f"  Momento linear total (deve ser ~0, pois o gerador zera o momento do CM):")
    print(f"    máximo |P(t)| (absoluto) ............... {momLin_max_abs:.3e}")
    print(f"    máximo |P(t)| relativo a m*v típico .... {momLin_max_rel:.3e}")
    print(f"  Momento angular total (deve ser constante):")
    print(f"    |L(0)| = {escala_L:.4e}")
    print(f"    desvio relativo máximo .................. {desvio_momAng_rel:.3e}")
    print(f"  Centro de massa (deve permanecer em ~(0,0,0)):")
    print(f"    deriva máxima |R_cm(t)| ................. {deriva_cm_max:.3e}")
    print("-" * 78)
    print("DISTÂNCIAS")
    for (i, j), d_ij in distancias.items():
        print(f"  corpo {i+1}-corpo {j+1}: mínima = {d_ij.min():.4f}   máxima = {d_ij.max():.4f}")
    print(f"  distância mínima global salva ........... {distancia_minima_global:.4f} "
          f"(limiar de corte de colisão é 1.0)")
    if motivoTermino == 1 and distancia_minima_global < 1.0:
        print("    [aviso] há um ponto salvo com distância < 1.0 mesmo em uma simulação")
        print("            marcada como cortada por colisão — reveja a MARGEM_SEGURANCA.")
    print("-" * 78)
    if erro_reversao is not None:
        print("TESTE DE REVERSIBILIDADE TEMPORAL (integra p/ frente e depois p/ trás)")
        print(f"  passos usados: {k}   erro relativo final ................ {erro_reversao:.3e}")
        print("  (um integrador simplético/reversível correto deve dar ~1e-10 a 1e-6;")
        print("   erros muito maiores sugerem bug na implementação do integrador)")
    else:
        print("TESTE DE REVERSIBILIDADE TEMPORAL: pulado (rode este script na pasta do")
        print("projeto, com fisica2corpos.py/fisica3corpos.py acessíveis, para habilitá-lo).")
    print("=" * 78)

    metricas = dict(
        arquivo=nome_arquivo, n_corpos=n_corpos, n_pontos=n_pontos,
        motivoTermino=motivoTermino, soma_massas=soma_massas,
        desvio_energia_rel=desvio_energia_rel, erro_energia_final_rel=erro_energia_final_rel,
        momLin_max_rel=momLin_max_rel, desvio_momAng_rel=desvio_momAng_rel,
        deriva_cm_max=deriva_cm_max, distancia_minima_global=distancia_minima_global,
        erro_bookkeeping_energia=erro_bookkeeping_energia,
        erro_bookkeeping_momLin=erro_bookkeeping_momLin,
        erro_bookkeeping_momAng=erro_bookkeeping_momAng,
        erro_reversao=erro_reversao, tem_nan_ou_inf=(tem_nan or tem_inf),
    )

    if gerar_grafico:
        fig = plotar_simulacao(d, distancias, erro_relativo_energia_t, R_cm_t, nome_arquivo)
        if salvar_fig:
            fig.savefig(salvar_fig, dpi=140, bbox_inches="tight")
            print(f"\nFigura salva em: {salvar_fig}")
        if mostrar_grafico:
            plt.show()
        else:
            plt.close(fig)

    return metricas


# ============================================================================
#                                  PLOTS
# ============================================================================

CORES = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange"]


def plotar_simulacao(d, distancias, erro_relativo_energia_t, R_cm_t, titulo):
    massas, estado, tempo = d["massas"], d["estado"], d["tempo"]
    momLin_salvo, momAng_salvo = d["momLin"], d["momAng"]
    n_corpos = len(massas)
    estado_sem_tempo = estado[:, :-1]

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f"Verificação da simulação — {titulo}", fontsize=13)

    # ---- 1. trajetória 3D --------------------------------------------------
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    for i in range(n_corpos):
        pos = estado_sem_tempo[:, 6 * i: 6 * i + 3]
        ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=CORES[i % len(CORES)],
                   label=f"corpo {i+1}", linewidth=1)
        ax3d.scatter(*pos[0], color=CORES[i % len(CORES)], marker="o", s=40)   # início
        ax3d.scatter(*pos[-1], color=CORES[i % len(CORES)], marker="X", s=60)  # fim
    ax3d.scatter(0, 0, 0, color="black", marker="+", s=80, label="origem (CM)")
    ax3d.set_title("Trajetória 3D (○ início, ✕ fim)")
    ax3d.legend(fontsize=7)

    # ---- 2. projeção XY -----------------------------------------------------
    axxy = fig.add_subplot(2, 3, 2)
    for i in range(n_corpos):
        pos = estado_sem_tempo[:, 6 * i: 6 * i + 3]
        axxy.plot(pos[:, 0], pos[:, 1], color=CORES[i % len(CORES)], label=f"corpo {i+1}")
        axxy.scatter(pos[0, 0], pos[0, 1], color=CORES[i % len(CORES)], marker="o")
        axxy.scatter(pos[-1, 0], pos[-1, 1], color=CORES[i % len(CORES)], marker="X")
    axxy.scatter(0, 0, color="black", marker="+", s=60)
    axxy.set_title("Projeção XY")
    axxy.set_xlabel("x")
    axxy.set_ylabel("y")
    axxy.axis("equal")
    axxy.grid(alpha=0.3)
    axxy.legend(fontsize=7)

    # ---- 3. distância entre pares -------------------------------------------
    axd = fig.add_subplot(2, 3, 3)
    for (i, j), d_ij in distancias.items():
        axd.plot(tempo, d_ij, label=f"{i+1}-{j+1}")
    axd.axhline(1.0, color="black", linestyle="--", linewidth=1, label="limiar de colisão")
    axd.set_title("Distância entre pares de corpos")
    axd.set_xlabel("tempo")
    axd.grid(alpha=0.3)
    axd.legend(fontsize=7)

    # ---- 4. erro relativo de energia ----------------------------------------
    axe = fig.add_subplot(2, 3, 4)
    axe.plot(tempo, erro_relativo_energia_t, color="tab:red")
    axe.set_title("Erro relativo de energia  (E(t)-E0)/|E0|")
    axe.set_xlabel("tempo")
    axe.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axe.grid(alpha=0.3)

    # ---- 5. momento linear ----------------------------------------------------
    axp = fig.add_subplot(2, 3, 5)
    axp.plot(tempo, momLin_salvo, color="tab:green")
    axp.set_title("|Momento linear total| (deve ficar ~0)")
    axp.set_xlabel("tempo")
    axp.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axp.grid(alpha=0.3)

    # ---- 6. desvio relativo do momento angular --------------------------------
    axl = fig.add_subplot(2, 3, 6)
    L0 = momAng_salvo[0]
    escala_L = max(np.linalg.norm(L0), 1e-12)
    desvio_L_t = np.linalg.norm(momAng_salvo - L0, axis=-1) / escala_L
    axl.plot(tempo, desvio_L_t, color="tab:purple")
    axl.set_title("Desvio relativo do momento angular")
    axl.set_xlabel("tempo")
    axl.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    axl.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ============================================================================
#                          MODO LOTE (pasta inteira)
# ============================================================================

def verificar_pasta(pasta, projeto, n_passos_reversao):
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.npz")))
    if not arquivos:
        print(f"Nenhum .npz encontrado em '{pasta}'.")
        return

    print(f"Verificando {len(arquivos)} arquivo(s) em '{pasta}' (modo resumido)...\n")
    linhas = []
    for caminho in arquivos:
        try:
            m = verificar_arquivo(caminho, projeto, n_passos_reversao=n_passos_reversao,
                                   gerar_grafico=False)
            linhas.append(m)
        except Exception as e:
            print(f"[erro] falha ao processar {caminho}: {e}")

    # tabela resumida ---------------------------------------------------------
    cab = f"{'arquivo':28s} {'N':>2s} {'pts':>6s} {'motivo':>6s} {'dE/E':>10s} {'|P|rel':>10s} {'dL/L':>10s} {'CM drift':>10s} {'d_min':>7s} {'reversão':>10s}"
    print("\n" + "=" * len(cab))
    print("RESUMO DO LOTE")
    print("=" * len(cab))
    print(cab)
    print("-" * len(cab))
    for m in linhas:
        rev = f"{m['erro_reversao']:.2e}" if m["erro_reversao"] is not None else "  n/d"
        print(f"{m['arquivo']:28s} {m['n_corpos']:2d} {m['n_pontos']:6d} {m['motivoTermino']:6d} "
              f"{m['desvio_energia_rel']:10.2e} {m['momLin_max_rel']:10.2e} {m['desvio_momAng_rel']:10.2e} "
              f"{m['deriva_cm_max']:10.2e} {m['distancia_minima_global']:7.3f} {rev:>10s}")
    print("=" * len(cab))

    problemas = [m for m in linhas if m["tem_nan_ou_inf"] or m["desvio_energia_rel"] > 1e-3]
    if problemas:
        print(f"\n[atenção] {len(problemas)} arquivo(s) com desvio de energia > 1e-3 e/ou NaN/Inf:")
        for m in problemas:
            print(f"  - {m['arquivo']}")
    else:
        print("\nNenhum arquivo com desvio de energia acima de 1e-3 ou NaN/Inf encontrado.")

    # detalha automaticamente o pior caso (maior desvio relativo de energia) --
    pior = max(linhas, key=lambda m: m["desvio_energia_rel"])
    caminho_pior = os.path.join(pasta, pior["arquivo"])
    print(f"\nDetalhando o pior caso (maior desvio de energia): {pior['arquivo']}\n")
    verificar_arquivo(caminho_pior, projeto, n_passos_reversao=n_passos_reversao,
                       gerar_grafico=True, mostrar_grafico=True)


# ============================================================================
#                                   MAIN
# ============================================================================

def encontrar_arquivo_mais_recente(projeto_dir):
    candidatos = []
    for pasta in ["simulacoesArtificiais/simulacoes2C", "simulacoesArtificiais/simulacoes3C"]:
        candidatos += glob.glob(os.path.join(projeto_dir, pasta, "*.npz"))
    if not candidatos:
        return None
    return max(candidatos, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Verifica simulações de N-corpos geradas pelo dataset.")
    parser.add_argument("arquivo", nargs="?", default=None,
                         help="Caminho de um .npz específico a verificar.")
    parser.add_argument("--pasta", default=None,
                         help="Verifica todos os .npz de uma pasta (modo resumido).")
    parser.add_argument("--projeto-dir", default=".",
                         help="Pasta com config.py/fisica2corpos.py/fisica3corpos.py (padrão: diretório atual).")
    parser.add_argument("--n-reversao", type=int, default=3000,
                         help="Nº de passos no teste de reversibilidade temporal (padrão: 3000).")
    parser.add_argument("--salvar-fig", default=None,
                         help="Salva a figura gerada neste caminho (ex: verificacao.png).")
    parser.add_argument("--sem-grafico", action="store_true",
                         help="Não exibe/gera gráfico, só o relatório impresso.")
    args = parser.parse_args()

    projeto = carregar_projeto(args.projeto_dir)

    if args.pasta:
        verificar_pasta(args.pasta, projeto, args.n_reversao)
        return

    caminho = args.arquivo
    if caminho is None:
        caminho = encontrar_arquivo_mais_recente(args.projeto_dir)
        if caminho is None:
            print("Nenhum arquivo .npz informado e nenhum encontrado em "
                  "simulacoesArtificiais/simulacoes2C ou simulacoes3C.")
            print("Use: python verificar_simulacao.py <caminho_do_arquivo.npz>")
            return
        print(f"Nenhum arquivo informado — verificando o mais recente: {caminho}\n")

    verificar_arquivo(caminho, projeto, n_passos_reversao=args.n_reversao,
                       gerar_grafico=not args.sem_grafico, salvar_fig=args.salvar_fig,
                       mostrar_grafico=not args.sem_grafico)


if __name__ == "__main__":
    main()