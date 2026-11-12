"""
Gerador de órbitas de 2 corpos por elementos orbitais (solução exata de Kepler).

Diferenças em relação ao geradorDataset.py:
  * Sem integração numérica: o estado é calculado analiticamente (energia e momento
    angular conservados até ~1e-15), então o dataset não carrega erro de integrador.
  * Unidades normalizadas: semieixo a = 1 e G*M = 1 (M = m1+m2 = 1). Toda órbita
    ligada vira a mesma família de 1 parâmetro (excentricidade e) + fase + orientação.
    O período é sempre 2*pi.
  * Amostragem por fase: PASSOS_POR_ORBITA amostras por período, igual para todas as
    órbitas. Portanto o dt entre amostras é constante (2*pi/PASSOS_POR_ORBITA) e o
    periapsis é resolvido de forma uniforme.
  * Cada trajetória cobre N_ORBITAS períodos completos, nada é cortado.
  * Excentricidade sorteada em [0, E_MAX] e periapsis a(1-e) >= 1 - E_MAX por construção,
    sem colisões e sem hiperbólicas.

Formato do .npz (compatível com carregarTrajetorias do saturn1b.py):
  estado  (N, 13): [r1(3), v1(3), r2(3), v2(3), t]   -> o treino usa só as 12 primeiras
  massas  (2,)
  tempo, energia, momAng, momLin
  dt      = dt entre amostras (use AMOSTRAGEM = 1 no treino e DT_EFETIVO = dt)
  a, e    = elementos orbitais (a = 1 no modo normalizado)

Uso:
  python geradorKepler.py --n 3000 --saida simulacoesArtificiais/simulacoesKepler
  python geradorKepler.py --n 400 --seed0 100000 --saida simulacoesArtificiais/kepler_val
"""
import argparse
import os
import numpy as np

GM = 1.0


def resolverKepler(M, e, tol=1e-14, maxIter=60):
    """Resolve E - e*sin(E) = M por Newton (vetorizado)."""
    M = np.mod(M, 2 * np.pi)
    E = np.where(e < 0.8, M, np.pi)
    for _ in range(maxIter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def rotacaoAleatoria(rng):
    """Matriz de rotação 3x3 uniforme em SO(3)."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def orbitaRelativa(a, e, M0, nPassos, dMedia):
    """Posição e velocidade relativas no plano orbital (G*M = 1)."""
    M = M0 + dMedia * np.arange(nPassos)
    E = resolverKepler(M, np.full_like(M, e))
    cE, sE = np.cos(E), np.sin(E)
    b = np.sqrt(1 - e**2)
    r = a * np.stack([cE - e, b * sE, np.zeros_like(E)], axis=1)
    fator = np.sqrt(GM / a) / (1 - e * cE)
    v = fator[:, None] * np.stack([-sE, b * cE, np.zeros_like(E)], axis=1)
    return r, v


def gerarTrajetoria(seed, passosPorOrbita=128, nOrbitas=8, eMax=0.7, massaMin=0.05):
    rng = np.random.default_rng(seed)
    a = 1.0
    e = rng.uniform(0.0, eMax)
    M0 = rng.uniform(0, 2 * np.pi)
    m1 = rng.uniform(massaMin, 1 - massaMin)
    m2 = 1.0 - m1

    nPassos = passosPorOrbita * nOrbitas
    dMedia = 2 * np.pi / passosPorOrbita          # n = sqrt(GM/a^3) = 1  ->  dt = dMedia
    r, v = orbitaRelativa(a, e, M0, nPassos, dMedia)

    R = rotacaoAleatoria(rng)
    r, v = r @ R.T, v @ R.T

    # centro de massa na origem, em repouso (M = 1)
    r1, v1 = -m2 * r, -m2 * v
    r2, v2 = m1 * r, m1 * v

    tempo = dMedia * np.arange(nPassos)
    estado = np.concatenate([r1, v1, r2, v2, tempo[:, None]], axis=1)

    rn = np.linalg.norm(r, axis=1)
    energia = 0.5 * m1 * (v1**2).sum(1) + 0.5 * m2 * (v2**2).sum(1) - m1 * m2 / rn
    momAng = m1 * np.cross(r1, v1) + m2 * np.cross(r2, v2)
    momLin = np.linalg.norm(m1 * v1 + m2 * v2, axis=1)
    return dict(massas=np.array([m1, m2]), estado=estado, tempo=tempo, energia=energia,
                momAng=momAng, momLin=momLin, dt=dMedia, motivoTermino=0, a=a, e=e)


# --------------------------------------------------------------------------------------
# Conversão entre o sistema físico e o normalizado (a = 1, GM = 1).
# Use para rodar a rede em trajetórias com qualquer escala (inclusive as do dataset antigo).
# --------------------------------------------------------------------------------------
def escalasDoSistema(estado, massas):
    """Retorna (L, tau): comprimento (semieixo) e tempo (T/2pi) do sistema. Exige E<0."""
    estado = np.asarray(estado)
    M = float(np.sum(massas))
    r = estado[..., 6:9] - estado[..., 0:3]
    v = estado[..., 9:12] - estado[..., 3:6]
    eps = 0.5 * (v**2).sum(-1) - GM * M / np.linalg.norm(r, axis=-1)
    eps = np.mean(eps)
    if eps >= 0:
        raise ValueError("órbita não ligada (E >= 0): não há semieixo")
    L = -GM * M / (2 * eps)
    tau = np.sqrt(L**3 / (GM * M))
    return L, tau


def paraNormalizado(estado, massas):
    """Estado físico (..., 12) -> unidades normalizadas. dt físico = dt_norm * tau."""
    L, tau = escalasDoSistema(estado, massas)
    out = np.array(estado[..., :12], dtype=float)
    for k in (0, 6):
        out[..., k:k + 3] /= L
        out[..., k + 3:k + 6] *= tau / L
    return out, L, tau


def deNormalizado(estado, L, tau):
    out = np.array(estado[..., :12], dtype=float)
    for k in (0, 6):
        out[..., k:k + 3] *= L
        out[..., k + 3:k + 6] *= L / tau
    return out


def _worker(args):
    seed, pasta, kw = args
    d = gerarTrajetoria(seed, **kw)
    np.savez_compressed(os.path.join(pasta, f"simulacaoKepler_{seed}.npz"), **d)
    return seed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--saida", default="simulacoesArtificiais/simulacoesKepler")
    ap.add_argument("--passosPorOrbita", type=int, default=128)
    ap.add_argument("--orbitas", type=int, default=8)
    ap.add_argument("--eMax", type=float, default=0.7)
    a = ap.parse_args()

    os.makedirs(a.saida, exist_ok=True)
    kw = dict(passosPorOrbita=a.passosPorOrbita, nOrbitas=a.orbitas, eMax=a.eMax)
    from multiprocessing import Pool
    with Pool(os.cpu_count()) as pool:
        pool.map(_worker, [(s, a.saida, kw) for s in range(a.seed0, a.seed0 + a.n)], chunksize=32)
    print(f"{a.n} trajetórias em {a.saida} | dt = {2*np.pi/a.passosPorOrbita:.5f} "
          f"| {a.passosPorOrbita*a.orbitas} passos por trajetória")
