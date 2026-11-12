'''import numpy as np, torch as t
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def carregarTrajetorias(pastaNPZ):
    trajetorias = []
    massasLista = []
    pasta = Path(pastaNPZ)
    for file in pasta.glob("*.npz"):
        dados = np.load(file)
        print(file,dados.files, dados["estado"].shape,dados["estado"].dtype) #deu problema pois a simulação 69 estava zerada 
        estados = dados["estado"][:, :18]
        trajetorias.append(estados)
        massasLista.append(dados["massas"])  # shape (3,)
    return trajetorias, massasLista


def calcularNormalizacao(traj):
    todos = np.concatenate(traj)
    media = np.mean(todos, axis=0)
    desvio = np.std(todos, axis=0)  # desvio padrão
    desvio = np.where(desvio < 1e-8, 1.0, desvio)  # evita divisão por zero em colunas constantes
    return media, desvio

def normalizar(estado, media, desvio):  # Z-SCORE
    return (estado - media) / desvio


trajetorias, massas = carregarTrajetorias("simulacoesArtificiais/simulacoes3C")

#dando um SHUFFLE nas massas e trajetórias mas mantendo o mesmo índice para que a rede tenha as massas corretas para cada simulação
indices = np.random.permutation(len(trajetorias))
trajetorias = [trajetorias[i] for i in indices]
massas = [massas[i] for i in indices]
nValidacao = int(len(trajetorias) / 5)  # 20% das trajetórias para VALIDAÇÃO
trajetoriasValidacao = trajetorias[:nValidacao]
trajetoriasTreino = trajetorias[nValidacao:]
massasValidacao = massas[:nValidacao]
massasTreino = massas[nValidacao:]

# normalização calculada SOMENTE com dados de treino (evita vazamento de dados)
media, desvio = calcularNormalizacao(trajetoriasTreino)

def calcularNormalizacaoMassa(massasLista):
    todas = np.stack(massasLista)  # (n_trajetorias, 3)
    media = np.mean(todas, axis=0)
    desvio = np.std(todas, axis=0)
    desvio = np.where(desvio < 1e-8, 1.0, desvio)
    return media, desvio

mediaMassa, desvioMassa = calcularNormalizacaoMassa(massasTreino)

# --- diagnóstico opcional (descomente se precisar checar novamente) ---
# print("desvio:", desvio)
# print("colunas com desvio zero ou quase zero:", np.where(desvio < 1e-8)[0])
# traj = trajetoriasValidacao[0]
# print("shape:", traj.shape)
# print("min/max por coluna:", traj.min(axis=0), traj.max(axis=0))


class Janelas(Dataset):
    def __init__(self, trajetorias, massas, tamanhoDaJanela, media, desvio, mediaMassa, desvioMassa):
        self.amostras = []
        for traj, massa in zip(trajetorias, massas):
            trajNormalizada = normalizar(traj, media, desvio)
            massaNorm = normalizar(massa, mediaMassa, desvioMassa)
            for i in range(len(trajNormalizada) - tamanhoDaJanela):
                janela = trajNormalizada[i:i + tamanhoDaJanela]
                alvo = trajNormalizada[i + tamanhoDaJanela]
                self.amostras.append((janela, massaNorm, alvo))

    def __len__(self):
        return len(self.amostras)

    def __getitem__(self, id):
        janela, massa, alvo = self.amostras[id]
        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massa, dtype=t.float32),
                t.tensor(alvo, dtype=t.float32))

TAMANHO_JANELA = 10

datasetTreino = Janelas(trajetoriasTreino, massasTreino, TAMANHO_JANELA, media, desvio, mediaMassa, desvioMassa)
datasetValidacao = Janelas(trajetoriasValidacao, massasValidacao, TAMANHO_JANELA, media, desvio, mediaMassa, desvioMassa)

loaderTreino = DataLoader(datasetTreino, batch_size=32, shuffle=True)
loaderValidacao = DataLoader(datasetValidacao, batch_size=32, shuffle=False)


class MLPBaseline(t.nn.Module):
    def __init__(self, tamanhoDaJanela, dimensaoDoEstado=18, dimensaoMassa=3, hidden=128):
        super().__init__()
        entrada = tamanhoDaJanela * dimensaoDoEstado + dimensaoMassa
        self.net = t.nn.Sequential(
            t.nn.Linear(entrada, hidden),
            t.nn.ReLU(),
            t.nn.Linear(hidden, hidden),
            t.nn.ReLU(),
            t.nn.Linear(hidden, dimensaoDoEstado)
        )

    def forward(self, janela, massa):
        x = janela.flatten(start_dim=1)
        x = t.cat([x, massa], dim=1)
        return self.net(x)


# IMPORTANDO O CUDA
device = t.device("cuda" if t.cuda.is_available() else "cpu")

modelo = MLPBaseline(tamanhoDaJanela=TAMANHO_JANELA, dimensaoDoEstado=18).to(device)
otimizador = t.optim.Adam(modelo.parameters(), lr=1e-3)
criterio = t.nn.MSELoss()

N_EPOCAS = 20
historicoTreino = []
historicoValidacao = []
melhorLossValidacao = float("inf")


def validar(modelo, loader, criterio, device):
    modelo.eval()
    lossAcumulada = 0.0
    with t.no_grad():
        for janela, massa, alvo in loader:
            janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)
            predicao = modelo(janela, massa)
            loss = criterio(predicao, alvo)
            lossAcumulada += loss.item()
    modelo.train()
    return lossAcumulada / len(loader)


def testarModelo(
    modelo,
    trajetoria,
    media,
    desvio,
    massa,
    mediaMassa, 
    desvioMassa,
    tamanhoDaJanela=10,
    device="cpu",
    nPassosRollout=1000
):
    """
    Avalia a MLP em uma trajetória nunca utilizada no treinamento,
    fazendo rollout AUTOREGRESSIVO: a cada passo, a predição do modelo
    é reinserida na janela para prever o próximo passo (em vez de
    sempre usar o dado real como entrada).

    trajetoria: array numpy (T, 18)
    media, desvio: usados na normalização
    nPassosRollout: quantidade de passos previstos autoregressivamente
    """

    modelo.eval()

    # =========================================================
    # 1. NORMALIZAÇÃO
    # =========================================================
    trajetoriaNorm = normalizar(trajetoria, media, desvio)
    T = len(trajetoriaNorm)
    nPassos = min(nPassosRollout, T - tamanhoDaJanela)

    # =========================================================
    # 2. PREVISÃO AUTOREGRESSIVA (ROLLOUT)
    # =========================================================
    previsoes = []
    alvos = []

    # janela inicial vem dos dados reais (ponto de partida obrigatório)
    janelaAtual = trajetoriaNorm[0:tamanhoDaJanela].copy()

    with t.no_grad():
        massaNorm = normalizar(massa, mediaMassa, desvioMassa)
        massaTensor = t.tensor(massaNorm, dtype=t.float32).unsqueeze(0).to(device)  
        for i in range(nPassos):
            alvo = trajetoriaNorm[i + tamanhoDaJanela]

            janelaTensor = t.tensor(
                janelaAtual, dtype=t.float32
            ).unsqueeze(0).to(device)

            pred = modelo(janelaTensor, massaTensor)
            pred = pred.squeeze(0).cpu().numpy()

            previsoes.append(pred)
            alvos.append(alvo)

            # desliza a janela: remove o estado mais antigo,
            # adiciona a PREDIÇÃO (não o dado real) no final
            janelaAtual = np.vstack([janelaAtual[1:], pred])

    previsoes = np.array(previsoes)
    alvos = np.array(alvos)

    # =========================================================
    # 3. DESNORMALIZAR
    # =========================================================
    previsoesFisicas = previsoes * desvio + media
    alvosFisicos = alvos * desvio + media

    # =========================================================
    # 4. ERROS
    # =========================================================
    erro = previsoesFisicas - alvosFisicos
    erroAbsoluto = np.abs(erro)
    mae = np.mean(erroAbsoluto)
    rmse = np.sqrt(np.mean(erro ** 2))
    erroMaximo = np.max(erroAbsoluto)

    print("\n" + "=" * 60)
    print("RESULTADO DA AVALIAÇÃO")
    print("=" * 60)
    print(f"Passos avaliados : {nPassos}")
    print(f"MAE              : {mae:.6e}")
    print(f"RMSE             : {rmse:.6e}")
    print(f"Erro máximo      : {erroMaximo:.6e}")

    # =========================================================
    # 5. SEPARAR POSIÇÃO E VELOCIDADE
    # =========================================================
    # corpo 1 -> 0:6 | corpo 2 -> 6:12 | corpo 3 -> 12:18
    # cada corpo: x y z vx vy vz
    posicoesReal, posicoesPred = [], []
    velocidadesReal, velocidadesPred = [], []

    for corpo in range(3):
        inicio = corpo * 6
        posicoesReal.append(alvosFisicos[:, inicio:inicio + 3])
        posicoesPred.append(previsoesFisicas[:, inicio:inicio + 3])
        velocidadesReal.append(alvosFisicos[:, inicio + 3:inicio + 6])
        velocidadesPred.append(previsoesFisicas[:, inicio + 3:inicio + 6])

    # =========================================================
    # 6. ERRO POR CORPO
    # =========================================================
    print("\nERRO POR CORPO")
    print("-" * 60)
    for corpo in range(3):
        erroPos = posicoesPred[corpo] - posicoesReal[corpo]
        erroVel = velocidadesPred[corpo] - velocidadesReal[corpo]
        rmsePos = np.sqrt(np.mean(erroPos ** 2))
        rmseVel = np.sqrt(np.mean(erroVel ** 2))
        print(f"Corpo {corpo + 1}: RMSE posição = {rmsePos:.6e} | RMSE velocidade = {rmseVel:.6e}")

    tempo = np.arange(nPassos)

    # =========================================================
    # 7. PLOT: POSIÇÃO REAL VS PREDITA
    # (eixo Y ajustado à escala do REAL, para o rollout divergente
    #  não "achatar" visualmente a curva real)
    # =========================================================
    for corpo in range(3):
        fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        nomes = ["X", "Y", "Z"]
        for eixo in range(3):
            real = posicoesReal[corpo][:, eixo]
            ax[eixo].plot(tempo, real, label="Real")
            ax[eixo].plot(tempo, posicoesPred[corpo][:, eixo], "--", label="Predito")
            ax[eixo].set_ylabel(nomes[eixo])
            ax[eixo].grid()
            ax[eixo].legend()

            margem = 0.2 * (real.max() - real.min() + 1e-8)
            ax[eixo].set_ylim(real.min() - margem, real.max() + margem)

        ax[2].set_xlabel("Passo da simulação")
        fig.suptitle(f"Corpo {corpo + 1} — Posição")
        plt.tight_layout()
        plt.show()

    # =========================================================
    # 8. PLOT: VELOCIDADE REAL VS PREDITA (mesmo ajuste de escala)
    # =========================================================
    for corpo in range(3):
        fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        nomes = ["Vx", "Vy", "Vz"]
        for eixo in range(3):
            real = velocidadesReal[corpo][:, eixo]
            ax[eixo].plot(tempo, real, label="Real")
            ax[eixo].plot(tempo, velocidadesPred[corpo][:, eixo], "--", label="Predito")
            ax[eixo].set_ylabel(nomes[eixo])
            ax[eixo].grid()
            ax[eixo].legend()

            margem = 0.2 * (real.max() - real.min() + 1e-8)
            ax[eixo].set_ylim(real.min() - margem, real.max() + margem)

        ax[2].set_xlabel("Passo da simulação")
        fig.suptitle(f"Corpo {corpo + 1} — Velocidade")
        plt.tight_layout()
        plt.show()

    # =========================================================
    # 9. ERRO DE POSIÇÃO AO LONGO DO TEMPO
    # =========================================================
    plt.figure(figsize=(10, 6))
    for corpo in range(3):
        erroPos = posicoesPred[corpo] - posicoesReal[corpo]
        normaErro = np.linalg.norm(erroPos, axis=1)
        plt.plot(tempo, normaErro, label=f"Corpo {corpo + 1}")
    plt.xlabel("Passo da simulação")
    plt.ylabel("Erro de posição")
    plt.title("Erro de posição ao longo do tempo (rollout)")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =========================================================
    # 10. ERRO DE VELOCIDADE
    # =========================================================
    plt.figure(figsize=(10, 6))
    for corpo in range(3):
        erroVel = velocidadesPred[corpo] - velocidadesReal[corpo]
        normaErro = np.linalg.norm(erroVel, axis=1)
        plt.plot(tempo, normaErro, label=f"Corpo {corpo + 1}")
    plt.xlabel("Passo da simulação")
    plt.ylabel("Erro de velocidade")
    plt.title("Erro de velocidade ao longo do tempo (rollout)")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =========================================================
    # 11. TRAJETÓRIA 3D
    # =========================================================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for corpo in range(3):
        ax.plot(posicoesReal[corpo][:, 0], posicoesReal[corpo][:, 1], posicoesReal[corpo][:, 2],
                label=f"Real C{corpo + 1}")
        ax.plot(posicoesPred[corpo][:, 0], posicoesPred[corpo][:, 1], posicoesPred[corpo][:, 2],
                "--", label=f"Predito C{corpo + 1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trajetória 3D — Real vs Predita")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # =========================================================
    # 12. RETORNO DOS RESULTADOS
    # =========================================================
    return {
        "mae": mae,
        "rmse": rmse,
        "erroMaximo": erroMaximo,
        "previsoes": previsoesFisicas,
        "alvos": alvosFisicos,
        "erro": erro
    }


# TREINO DO MODELO
for epoca in range(1, N_EPOCAS + 1):
    modelo.train()
    loss_acumulada = 0.0

    for janela, massa, alvo in loaderTreino:
        janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)
        pred = modelo(janela, massa)
        loss = criterio(pred, alvo)

        otimizador.zero_grad()
        loss.backward()
        otimizador.step()

        loss_acumulada += loss.item()

    loss_treino = loss_acumulada / len(loaderTreino)
    loss_val = validar(modelo, loaderValidacao, criterio, device)

    historicoTreino.append(loss_treino)
    historicoValidacao.append(loss_val)

    print(f"Época {epoca:3d} | treino: {loss_treino:.6f} | val: {loss_val:.6f}")

    # salva só quando a validação melhora (evita levar overfitting pro arquivo final)
    if loss_val < melhorLossValidacao:
        melhorLossValidacao = loss_val
        t.save({
            "pesos_modelo": modelo.state_dict(),
            "media": media,
            "desvio": desvio,
            "mediaMassa": mediaMassa,      # <- adicionar
            "desvioMassa": desvioMassa,    # <- adicionar
            "k": TAMANHO_JANELA,
            "epoca": epoca,
        }, "baseline_mlp.pt")

# recarrega o MELHOR checkpoint salvo (não o modelo da última época)
checkpoint = t.load("baseline_mlp.pt", map_location=device, weights_only=False)
modelo.load_state_dict(checkpoint["pesos_modelo"])
print(f"Carregado checkpoint da época {checkpoint['epoca']} (melhor val loss)")

resultado = testarModelo(
    modelo=modelo,
    trajetoria=trajetoriasValidacao[0],
    media=media,
    desvio=desvio,
    massa=massasValidacao[0],
    mediaMassa=mediaMassa, 
    desvioMassa=desvioMassa,
    tamanhoDaJanela=TAMANHO_JANELA,
    device=device,
    nPassosRollout=1000
)

print("Real - min/max:", resultado["alvos"].min(), resultado["alvos"].max())
print("Pred - min/max:", resultado["previsoes"].min(), resultado["previsoes"].max())
print("Real - std:", resultado["alvos"].std())
print("Pred - std:", resultado["previsoes"].std())'''
'''import numpy as np, torch as t
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =====================================================================
# RESUMO DAS ALTERAÇÕES EM RELAÇÃO AO SCRIPT ORIGINAL
# =====================================================================
# 1) DATA AUGMENTATION POR SIMETRIA FÍSICA
#    A gravidade é invariante por rotação: rotacionar posições E
#    velocidades dos 3 corpos pela MESMA matriz de rotação gera uma
#    trajetória igualmente válida. Isso multiplica a diversidade
#    efetiva do dataset sem precisar de novas simulações.
#
# 2) SCHEDULED SAMPLING (mistura de teacher forcing e rollout no treino)
#    Em vez de treinar só com janelas de dados reais, a rede agora
#    "desenrola" HORIZONTE_TREINO passos por amostra durante o treino,
#    e a cada passo decide (com probabilidade p, que decai com as
#    épocas) se alimenta a janela seguinte com o dado real ou com a
#    própria predição. Isso aproxima a distribuição vista no treino da
#    distribuição vista no rollout autoregressivo do teste.
#
# 3) REGULARIZAÇÃO
#    Dropout nas camadas ocultas + weight_decay no Adam, para segurar
#    o overfitting precoce observado (val loss piorando já na época 2).
#
# 4) MÉTRICAS MAIS REALISTAS PARA SISTEMA CAÓTICO
#    Além de MAE/RMSE globais, calculamos: (a) "horizonte de previsão
#    útil" — quantos passos até o erro relativo de posição ultrapassar
#    limiares (10%/50%/100%); (b) deriva do momento linear e angular
#    do rollout predito (grandezas que a física exige que se conservem,
#    checáveis sem precisar da constante G).
#
# A previsão continua sendo do ESTADO ABSOLUTO no próximo passo (não
# do delta), como solicitado — só o *protocolo de treino* e a
# *avaliação* mudaram.
#
# CORREÇÃO (após 1ª rodada de testes): a normalização por coluna
# (um desvio-padrão por x/y/z) é incompatível com a augmentation por
# rotação, pois rotacionar mistura x, y, z mas cada um continua sendo
# dividido pelo desvio ORIGINAL daquela coluna — isso inflou a loss em
# ~60x e destruiu a conservação de momento no rollout. Trocado por uma
# normalização ISOTRÓPICA (um desvio compartilhado para toda posição,
# outro para toda velocidade, média zero) em `calcularNormalizacaoEstado`.
# =====================================================================


def carregarTrajetorias(pastaNPZ):
    trajetorias = []
    massasLista = []
    pasta = Path(pastaNPZ)
    for file in sorted(pasta.glob("*.npz")):
        dados = np.load(file)
        estado = dados["estado"]
        if estado.shape[0] == 0:
            # simulação que falhou e ficou vazia (ex.: simulacao3D_69.npz) — pula
            print(f"[aviso] pulando {file.name}: trajetória vazia")
            continue
        estados = estado[:, :18]
        trajetorias.append(estados)
        massasLista.append(dados["massas"])  # shape (3,)
    return trajetorias, massasLista


def calcularNormalizacaoEstado(trajetorias):
    """
    Normalização ISOTRÓPICA POR CORPO: x, y, z de um mesmo corpo
    dividem pelo MESMO desvio (necessário para ser compatível com a
    augmentation por rotação), mas cada CORPO tem seu próprio desvio,
    tanto para posição quanto para velocidade.

    Por que não um único desvio para os 3 corpos juntos (1ª tentativa):
    corpos diferentes costumam ter amplitudes de movimento bem
    diferentes (o corpo mais massivo se move menos, por conservação de
    momento). Um único desvio "médio" comprime demais o corpo que se
    move pouco e amplia demais o que se move muito, prejudicando o
    condicionamento do problema de regressão. Usar um desvio por corpo
    mantém a normalização próxima da z-score original coluna-a-coluna
    (que funcionava bem) — mas ainda respeitando x,y,z do MESMO corpo
    juntos, que é o único requisito real para compatibilidade com a
    rotação (a rotação nunca mistura dados de corpos diferentes entre
    si, então nada impede escalas diferentes entre corpos).

    Média zero: em valor esperado, rotações uniformes aleatórias em
    torno da origem já centralizam os dados aumentados em zero.
    """
    todos = np.concatenate(trajetorias)      # (N, 18)
    bloco = todos.reshape(-1, 3, 6)          # (N, corpo, 6)

    desvio = np.zeros(18)
    for corpo in range(3):
        posCorpo = bloco[:, corpo, 0:3]      # (N, 3) — x,y,z desse corpo
        velCorpo = bloco[:, corpo, 3:6]      # (N, 3) — vx,vy,vz desse corpo

        desvioPosCorpo = np.std(posCorpo)
        desvioVelCorpo = np.std(velCorpo)
        desvioPosCorpo = desvioPosCorpo if desvioPosCorpo > 1e-8 else 1.0
        desvioVelCorpo = desvioVelCorpo if desvioVelCorpo > 1e-8 else 1.0

        inicio = corpo * 6
        desvio[inicio:inicio + 3] = desvioPosCorpo
        desvio[inicio + 3:inicio + 6] = desvioVelCorpo

    media = np.zeros(18)
    return media, desvio


def normalizar(estado, media, desvio):  # Z-SCORE
    return (estado - media) / desvio


def calcularNormalizacaoMassa(massasLista):
    todas = np.stack(massasLista)  # (n_trajetorias, 3)
    media = np.mean(todas, axis=0)
    desvio = np.std(todas, axis=0)
    desvio = np.where(desvio < 1e-8, 1.0, desvio)
    return media, desvio


# =====================================================================
# ALTERAÇÃO 1 — AUGMENTATION POR ROTAÇÃO (simetria física da gravidade)
# =====================================================================
def gerarRotacaoAleatoria():
    """
    Gera uma matriz de rotação 3D uniformemente distribuída em SO(3)
    (algoritmo de Mezzadri: QR de uma matriz gaussiana aleatória).
    """
    A = np.random.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    D = np.diag(np.sign(np.diag(R)))
    Q = Q @ D
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def aplicarRotacaoEstado(estado, R):
    """
    Aplica a MESMA rotação R às posições e velocidades dos 3 corpos.
    estado: array (..., 18) em unidades FÍSICAS (não normalizado).
    Layout por corpo: [x, y, z, vx, vy, vz] -> reshape (..., 3 corpos, 6).
    """
    forma = estado.shape[:-1] + (3, 6)
    bloco = estado.reshape(forma).copy()
    bloco[..., 0:3] = bloco[..., 0:3] @ R.T  # posições
    bloco[..., 3:6] = bloco[..., 3:6] @ R.T  # velocidades
    return bloco.reshape(estado.shape)


trajetorias, massas = carregarTrajetorias("simulacoesArtificiais/simulacoes3C")

# dando um SHUFFLE nas massas e trajetórias mas mantendo o mesmo índice para que a rede tenha as massas corretas para cada simulação
indices = np.random.permutation(len(trajetorias))
trajetorias = [trajetorias[i] for i in indices]
massas = [massas[i] for i in indices]
nValidacao = int(len(trajetorias) / 5)  # 20% das trajetórias para VALIDAÇÃO
trajetoriasValidacao = trajetorias[:nValidacao]
trajetoriasTreino = trajetorias[nValidacao:]
massasValidacao = massas[:nValidacao]
massasTreino = massas[nValidacao:]

# normalização ISOTRÓPICA calculada SOMENTE com dados de treino
# (compatível com a augmentation por rotação — ver docstring da função)
media, desvio = calcularNormalizacaoEstado(trajetoriasTreino)
mediaMassa, desvioMassa = calcularNormalizacaoMassa(massasTreino)


# =====================================================================
# ALTERAÇÃO 2 — DATASET COM JANELA + HORIZONTE (para scheduled sampling)
# =====================================================================
class JanelasRollout(Dataset):
    """
    Cada amostra fornece:
      - janela: K passos de contexto (normalizados)
      - massa:  massas normalizadas dos 3 corpos
      - futuros: H passos futuros reais (normalizados), usados como alvo
                 em cada um dos H passos do rollout de treino

    Guarda as trajetórias em unidades FÍSICAS (não normalizadas) e só
    normaliza (após rotacionar, se augmentar=True) dentro de
    __getitem__, para que a rotação seja aplicada nos dados físicos.
    """

    def __init__(self, trajetorias, massas, tamanhoDaJanela, horizonte,
                 media, desvio, mediaMassa, desvioMassa, augmentar=False):
        self.trajetorias = trajetorias
        self.massas = massas
        self.tamanhoDaJanela = tamanhoDaJanela
        self.horizonte = horizonte
        self.media = media
        self.desvio = desvio
        self.mediaMassa = mediaMassa
        self.desvioMassa = desvioMassa
        self.augmentar = augmentar

        comprimentoTotal = tamanhoDaJanela + horizonte
        self.indices = []  # (idxTrajetoria, posicaoInicial)
        for idxTraj, traj in enumerate(trajetorias):
            nAmostras = len(traj) - comprimentoTotal + 1
            for i in range(max(nAmostras, 0)):
                self.indices.append((idxTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, id):
        idxTraj, i = self.indices[id]
        traj = self.trajetorias[idxTraj]
        massa = self.massas[idxTraj]
        comprimentoTotal = self.tamanhoDaJanela + self.horizonte

        bloco = traj[i:i + comprimentoTotal].copy()  # (K+H, 18) em unidades físicas

        if self.augmentar:
            R = gerarRotacaoAleatoria()
            bloco = aplicarRotacaoEstado(bloco, R)

        blocoNorm = normalizar(bloco, self.media, self.desvio)
        janela = blocoNorm[:self.tamanhoDaJanela]
        futuros = blocoNorm[self.tamanhoDaJanela:]  # (H, 18)

        massaNorm = normalizar(massa, self.mediaMassa, self.desvioMassa)

        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massaNorm, dtype=t.float32),
                t.tensor(futuros, dtype=t.float32))


TAMANHO_JANELA = 10
HORIZONTE_TREINO = 5  # quantos passos desenrolar durante o treino (scheduled sampling)

# treino: com augmentation por rotação e horizonte multi-passo
datasetTreino = JanelasRollout(trajetoriasTreino, massasTreino, TAMANHO_JANELA,
                                HORIZONTE_TREINO, media, desvio, mediaMassa, desvioMassa,
                                augmentar=True)
# validação: sem augmentation, horizonte=1 (loss de 1 passo só, para critério de "melhor época" comparável ao original)
datasetValidacao = JanelasRollout(trajetoriasValidacao, massasValidacao, TAMANHO_JANELA,
                                   1, media, desvio, mediaMassa, desvioMassa,
                                   augmentar=False)

loaderTreino = DataLoader(datasetTreino, batch_size=32, shuffle=True)
loaderValidacao = DataLoader(datasetValidacao, batch_size=32, shuffle=False)


# =====================================================================
# ALTERAÇÃO 3 — REGULARIZAÇÃO (dropout no modelo + weight_decay no Adam)
# =====================================================================
class MLPBaseline(t.nn.Module):
    def __init__(self, tamanhoDaJanela, dimensaoDoEstado=18, dimensaoMassa=3, hidden=128, dropout=0.1):
        super().__init__()
        entrada = tamanhoDaJanela * dimensaoDoEstado + dimensaoMassa
        self.net = t.nn.Sequential(
            t.nn.Linear(entrada, hidden),
            t.nn.ReLU(),
            t.nn.Dropout(dropout),
            t.nn.Linear(hidden, hidden),
            t.nn.ReLU(),
            t.nn.Dropout(dropout),
            t.nn.Linear(hidden, dimensaoDoEstado)
        )

    def forward(self, janela, massa):
        x = janela.flatten(start_dim=1)
        x = t.cat([x, massa], dim=1)
        return self.net(x)


# IMPORTANDO O CUDA
device = t.device("cuda" if t.cuda.is_available() else "cpu")

modelo = MLPBaseline(tamanhoDaJanela=TAMANHO_JANELA, dimensaoDoEstado=18, dropout=0.1).to(device)
otimizador = t.optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
criterio = t.nn.MSELoss()

N_EPOCAS = 20
historicoTreino = []
historicoValidacao = []
melhorLossValidacao = float("inf")


def probabilidadeTeacherForcing(epoca, nEpocasTotais, pInicial=1.0, pFinal=0.05):
    """Decai linearmente de pInicial (só dado real) até pFinal (quase só rollout)."""
    if nEpocasTotais <= 1:
        return pFinal
    frac = (epoca - 1) / (nEpocasTotais - 1)
    frac = min(max(frac, 0.0), 1.0)
    return pInicial + (pFinal - pInicial) * frac


def validar(modelo, loader, criterio, device):
    modelo.eval()
    lossAcumulada = 0.0
    with t.no_grad():
        for janela, massa, futuros in loader:
            janela, massa, futuros = janela.to(device), massa.to(device), futuros.to(device)
            alvo = futuros[:, 0, :]
            predicao = modelo(janela, massa)
            loss = criterio(predicao, alvo)
            lossAcumulada += loss.item()
    modelo.train()
    return lossAcumulada / len(loader)


# =====================================================================
# ALTERAÇÃO 4 — MÉTRICAS FÍSICAS PARA SISTEMA CAÓTICO
# =====================================================================
def calcularMomentoLinear(estadoFisico, massas):
    """estadoFisico: (..., 18). Retorna momento linear total (..., 3). Não precisa de G."""
    forma = estadoFisico.shape[:-1] + (3, 6)
    estado = estadoFisico.reshape(forma)
    vel = estado[..., 3:6]
    massasArr = np.asarray(massas).reshape(3, 1)
    return np.sum(vel * massasArr, axis=-2)


def calcularMomentoAngular(estadoFisico, massas):
    """estadoFisico: (..., 18). Retorna momento angular total (..., 3). Não precisa de G."""
    forma = estadoFisico.shape[:-1] + (3, 6)
    estado = estadoFisico.reshape(forma)
    pos = estado[..., 0:3]
    vel = estado[..., 3:6]
    massasArr = np.asarray(massas).reshape(3, 1)
    L = np.cross(pos, vel) * massasArr
    return np.sum(L, axis=-2)


def calcularHorizontesPrevisao(previsoesFisicas, alvosFisicos, limiares=(0.1, 0.5, 1.0)):
    """
    Para cada limiar de erro relativo de posição, retorna em quantos
    passos o rollout ultrapassa esse limiar. Métrica mais informativa
    que MAE/RMSE em passo fixo para sistemas caóticos, onde o erro
    cresce exponencialmente após um certo horizonte.
    """
    nPassos = previsoesFisicas.shape[0]
    posReal = alvosFisicos.reshape(nPassos, 3, 6)[..., :3]
    posPred = previsoesFisicas.reshape(nPassos, 3, 6)[..., :3]
    erroPorPasso = np.linalg.norm(posPred - posReal, axis=-1).sum(axis=-1)
    refPorPasso = np.linalg.norm(posReal, axis=-1).sum(axis=-1)
    erroRelativo = erroPorPasso / (refPorPasso + 1e-8)

    resultados = {}
    for limiar in limiares:
        acima = np.where(erroRelativo > limiar)[0]
        resultados[limiar] = int(acima[0]) if len(acima) > 0 else nPassos
    return resultados


def testarModelo(
    modelo,
    trajetoria,
    media,
    desvio,
    massa,
    mediaMassa,
    desvioMassa,
    tamanhoDaJanela=10,
    device="cpu",
    nPassosRollout=1000
):
    """
    Avalia a MLP em uma trajetória nunca utilizada no treinamento,
    fazendo rollout AUTOREGRESSIVO: a cada passo, a predição do modelo
    é reinserida na janela para prever o próximo passo.
    """

    modelo.eval()

    trajetoriaNorm = normalizar(trajetoria, media, desvio)
    T = len(trajetoriaNorm)
    nPassos = min(nPassosRollout, T - tamanhoDaJanela)

    previsoes = []
    alvos = []
    janelaAtual = trajetoriaNorm[0:tamanhoDaJanela].copy()

    with t.no_grad():
        massaNorm = normalizar(massa, mediaMassa, desvioMassa)
        massaTensor = t.tensor(massaNorm, dtype=t.float32).unsqueeze(0).to(device)
        for i in range(nPassos):
            alvo = trajetoriaNorm[i + tamanhoDaJanela]

            janelaTensor = t.tensor(janelaAtual, dtype=t.float32).unsqueeze(0).to(device)

            pred = modelo(janelaTensor, massaTensor)
            pred = pred.squeeze(0).cpu().numpy()

            previsoes.append(pred)
            alvos.append(alvo)

            janelaAtual = np.vstack([janelaAtual[1:], pred])

    previsoes = np.array(previsoes)
    alvos = np.array(alvos)

    previsoesFisicas = previsoes * desvio + media
    alvosFisicos = alvos * desvio + media

    erro = previsoesFisicas - alvosFisicos
    erroAbsoluto = np.abs(erro)
    mae = np.mean(erroAbsoluto)
    rmse = np.sqrt(np.mean(erro ** 2))
    erroMaximo = np.max(erroAbsoluto)

    print("\n" + "=" * 60)
    print("RESULTADO DA AVALIAÇÃO")
    print("=" * 60)
    print(f"Passos avaliados : {nPassos}")
    print(f"MAE              : {mae:.6e}")
    print(f"RMSE             : {rmse:.6e}")
    print(f"Erro máximo      : {erroMaximo:.6e}")

    # ---- métrica 1: horizonte de previsão útil (sistema caótico) ----
    horizontes = calcularHorizontesPrevisao(previsoesFisicas, alvosFisicos, limiares=(0.1, 0.5, 1.0))
    print("\nHORIZONTE DE PREVISÃO ÚTIL (passos até o erro relativo de posição ultrapassar o limiar)")
    print("-" * 60)
    for limiar, horizonte in horizontes.items():
        print(f"limiar {int(limiar * 100):3d}% de erro relativo: {horizonte} passos")

    # ---- métrica 2: conservação de momento linear / angular ----
    momentoPred = calcularMomentoLinear(previsoesFisicas, massa)
    momentoReal = calcularMomentoLinear(alvosFisicos, massa)
    angularPred = calcularMomentoAngular(previsoesFisicas, massa)
    angularReal = calcularMomentoAngular(alvosFisicos, massa)

    derivaMomentoPred = np.linalg.norm(momentoPred - momentoPred[0], axis=1)
    derivaAngularPred = np.linalg.norm(angularPred - angularPred[0], axis=1)
    derivaMomentoReal = np.linalg.norm(momentoReal - momentoReal[0], axis=1)
    derivaAngularReal = np.linalg.norm(angularReal - angularReal[0], axis=1)

    print("\nCONSERVAÇÃO DE MOMENTO (não depende de G, só das massas)")
    print("-" * 60)
    print(f"[sanity-check simulador] deriva máx. momento linear REAL : {derivaMomentoReal.max():.3e}")
    print(f"[sanity-check simulador] deriva máx. momento angular REAL: {derivaAngularReal.max():.3e}")
    print(f"deriva máx. momento linear PREDITO : {derivaMomentoPred.max():.3e}")
    print(f"deriva máx. momento angular PREDITO: {derivaAngularPred.max():.3e}")

    # ---- erro por corpo (mantido do original) ----
    posicoesReal, posicoesPred = [], []
    velocidadesReal, velocidadesPred = [], []
    for corpo in range(3):
        inicio = corpo * 6
        posicoesReal.append(alvosFisicos[:, inicio:inicio + 3])
        posicoesPred.append(previsoesFisicas[:, inicio:inicio + 3])
        velocidadesReal.append(alvosFisicos[:, inicio + 3:inicio + 6])
        velocidadesPred.append(previsoesFisicas[:, inicio + 3:inicio + 6])

    print("\nERRO POR CORPO")
    print("-" * 60)
    for corpo in range(3):
        erroPos = posicoesPred[corpo] - posicoesReal[corpo]
        erroVel = velocidadesPred[corpo] - velocidadesReal[corpo]
        rmsePos = np.sqrt(np.mean(erroPos ** 2))
        rmseVel = np.sqrt(np.mean(erroVel ** 2))
        print(f"Corpo {corpo + 1}: RMSE posição = {rmsePos:.6e} | RMSE velocidade = {rmseVel:.6e}")

    tempo = np.arange(nPassos)

    # ---- plots de posição/velocidade (mantidos do original) ----
    for corpo in range(3):
        fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        nomes = ["X", "Y", "Z"]
        for eixo in range(3):
            real = posicoesReal[corpo][:, eixo]
            ax[eixo].plot(tempo, real, label="Real")
            ax[eixo].plot(tempo, posicoesPred[corpo][:, eixo], "--", label="Predito")
            ax[eixo].set_ylabel(nomes[eixo])
            ax[eixo].grid()
            ax[eixo].legend()
            margem = 0.2 * (real.max() - real.min() + 1e-8)
            ax[eixo].set_ylim(real.min() - margem, real.max() + margem)
        ax[2].set_xlabel("Passo da simulação")
        fig.suptitle(f"Corpo {corpo + 1} — Posição")
        plt.tight_layout()
        plt.show()

    for corpo in range(3):
        fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        nomes = ["Vx", "Vy", "Vz"]
        for eixo in range(3):
            real = velocidadesReal[corpo][:, eixo]
            ax[eixo].plot(tempo, real, label="Real")
            ax[eixo].plot(tempo, velocidadesPred[corpo][:, eixo], "--", label="Predito")
            ax[eixo].set_ylabel(nomes[eixo])
            ax[eixo].grid()
            ax[eixo].legend()
            margem = 0.2 * (real.max() - real.min() + 1e-8)
            ax[eixo].set_ylim(real.min() - margem, real.max() + margem)
        ax[2].set_xlabel("Passo da simulação")
        fig.suptitle(f"Corpo {corpo + 1} — Velocidade")
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(10, 6))
    for corpo in range(3):
        erroPos = posicoesPred[corpo] - posicoesReal[corpo]
        normaErro = np.linalg.norm(erroPos, axis=1)
        plt.plot(tempo, normaErro, label=f"Corpo {corpo + 1}")
    plt.xlabel("Passo da simulação")
    plt.ylabel("Erro de posição")
    plt.title("Erro de posição ao longo do tempo (rollout)")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for corpo in range(3):
        erroVel = velocidadesPred[corpo] - velocidadesReal[corpo]
        normaErro = np.linalg.norm(erroVel, axis=1)
        plt.plot(tempo, normaErro, label=f"Corpo {corpo + 1}")
    plt.xlabel("Passo da simulação")
    plt.ylabel("Erro de velocidade")
    plt.title("Erro de velocidade ao longo do tempo (rollout)")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for corpo in range(3):
        ax.plot(posicoesReal[corpo][:, 0], posicoesReal[corpo][:, 1], posicoesReal[corpo][:, 2],
                label=f"Real C{corpo + 1}")
        ax.plot(posicoesPred[corpo][:, 0], posicoesPred[corpo][:, 1], posicoesPred[corpo][:, 2],
                "--", label=f"Predito C{corpo + 1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trajetória 3D — Real vs Predita")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # ---- novo plot: deriva das grandezas conservadas ----
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(tempo, derivaMomentoPred, label="Predito")
    ax[0].axhline(0, color="gray", linestyle=":", label="Real (conservado)")
    ax[0].set_ylabel("‖p(t) - p(0)‖")
    ax[0].set_title("Conservação do momento linear (rollout)")
    ax[0].set_yscale("log")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(tempo, derivaAngularPred, label="Predito")
    ax[1].axhline(0, color="gray", linestyle=":", label="Real (conservado)")
    ax[1].set_ylabel("‖L(t) - L(0)‖")
    ax[1].set_xlabel("Passo da simulação")
    ax[1].set_title("Conservação do momento angular (rollout)")
    ax[1].set_yscale("log")
    ax[1].grid()
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    return {
        "mae": mae,
        "rmse": rmse,
        "erroMaximo": erroMaximo,
        "horizontesPrevisao": horizontes,
        "derivaMomentoPredMax": derivaMomentoPred.max(),
        "derivaAngularPredMax": derivaAngularPred.max(),
        "previsoes": previsoesFisicas,
        "alvos": alvosFisicos,
        "erro": erro
    }


# =====================================================================
# TREINO DO MODELO (com scheduled sampling)
# =====================================================================
for epoca in range(1, N_EPOCAS + 1):
    modelo.train()
    loss_acumulada = 0.0
    p = probabilidadeTeacherForcing(epoca, N_EPOCAS)

    for janela, massa, futuros in loaderTreino:
        janela = janela.to(device)      # (B, K, 18)
        massa = massa.to(device)        # (B, 3)
        futuros = futuros.to(device)    # (B, H, 18)

        otimizador.zero_grad()

        janelaAtual = janela
        lossPassos = []
        for h in range(HORIZONTE_TREINO):
            alvo = futuros[:, h, :]
            pred = modelo(janelaAtual, massa)
            lossPassos.append(criterio(pred, alvo))

            # decide, por amostra do batch, se o próximo passo da janela
            # usa o dado real (teacher forcing) ou a própria predição
            usarReal = (t.rand(janelaAtual.size(0), 1, device=device) < p).float()
            proximoEstado = usarReal * alvo + (1 - usarReal) * pred.detach()

            janelaAtual = t.cat([janelaAtual[:, 1:, :], proximoEstado.unsqueeze(1)], dim=1)

        loss = t.stack(lossPassos).mean()
        loss.backward()
        t.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)  # segurança extra p/ rollout multi-passo
        otimizador.step()

        loss_acumulada += loss.item()

    loss_treino = loss_acumulada / len(loaderTreino)
    loss_val = validar(modelo, loaderValidacao, criterio, device)

    historicoTreino.append(loss_treino)
    historicoValidacao.append(loss_val)

    print(f"Época {epoca:3d} | p_teacher_forcing: {p:.2f} | treino: {loss_treino:.6f} | val: {loss_val:.6f}")

    if loss_val < melhorLossValidacao:
        melhorLossValidacao = loss_val
        t.save({
            "pesos_modelo": modelo.state_dict(),
            "media": media,
            "desvio": desvio,
            "mediaMassa": mediaMassa,
            "desvioMassa": desvioMassa,
            "k": TAMANHO_JANELA,
            "horizonteTreino": HORIZONTE_TREINO,
            "epoca": epoca,
        }, "baseline_mlp.pt")

checkpoint = t.load("baseline_mlp.pt", map_location=device, weights_only=False)
modelo.load_state_dict(checkpoint["pesos_modelo"])
print(f"Carregado checkpoint da época {checkpoint['epoca']} (melhor val loss)")

resultado = testarModelo(
    modelo=modelo,
    trajetoria=trajetoriasValidacao[0],
    media=media,
    desvio=desvio,
    massa=massasValidacao[0],
    mediaMassa=mediaMassa,
    desvioMassa=desvioMassa,
    tamanhoDaJanela=TAMANHO_JANELA,
    device=device,
    nPassosRollout=1000
)

print("Real - min/max:", resultado["alvos"].min(), resultado["alvos"].max())
print("Pred - min/max:", resultado["previsoes"].min(), resultado["previsoes"].max())
print("Real - std:", resultado["alvos"].std())
print("Pred - std:", resultado["previsoes"].std())'''

"""import numpy as np
import torch as t
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =====================================================================
# EXEMPLO MÍNIMO: MLP que prevê o PRÓXIMO ESTADO do problema de 3 corpos
# =====================================================================
# Objetivo: mostrar a estrutura MÍNIMA necessária para uma rede neural
# aprender a prever o próximo estado (posição+velocidade dos 3 corpos)
# a partir de uma janela de estados passados.
#
# O que foi DELIBERADAMENTE OMITIDO em relação ao script completo
# (isso é refinamento, não requisito):
#   - data augmentation por rotação
#   - scheduled sampling / rollout multi-passo no treino
#   - dropout / weight_decay
#   - métricas físicas (momento linear/angular, horizonte de previsão)
#   - previsão do Hamiltoniano (fica pra depois, como você pediu)
#
# Estrutura mínima = 5 blocos, e é só isso que uma rede precisa pra
# "funcionar" nesse problema:
#   1. Carregar dados (trajetórias simuladas por um integrador)
#   2. Normalizar (z-score) — sem isso o treino não converge bem,
#      pois posição e velocidade têm escalas muito diferentes
#   3. Dataset: janela de K passos passados -> 1 passo futuro
#   4. Modelo: MLP simples (poucas camadas densas + ReLU)
#   5. Loop de treino: forward -> loss (MSE) -> backward -> step
# =====================================================================


# ---------------------------------------------------------------
# 1. CARREGAR DADOS
# ---------------------------------------------------------------
def carregarTrajetorias(pastaNPZ):
    
    trajetorias, massasLista = [], []
    for arquivo in sorted(Path(pastaNPZ).glob("*.npz")):
        dados = np.load(arquivo)
        estado = dados["estado"][:, :18]
        if estado.shape[0] == 0:
            continue
        trajetorias.append(estado)
        massasLista.append(dados["massas"])
    return trajetorias, massasLista


# ---------------------------------------------------------------
# 2. NORMALIZAÇÃO (z-score simples, coluna a coluna)
# ---------------------------------------------------------------
def calcularNormalizacao(trajetorias):
    todos = np.concatenate(trajetorias)          # (N, 18)
    media = todos.mean(axis=0)
    desvio = todos.std(axis=0)
    desvio[desvio < 1e-8] = 1.0
    return media, desvio


def normalizar(x, media, desvio):
    return (x - media) / desvio


def desnormalizar(x, media, desvio):
    return x * desvio + media


# ---------------------------------------------------------------
# 3. DATASET: janela de K passos -> próximo passo
# ---------------------------------------------------------------
class JanelaProximoEstado(Dataset):
    
    def __init__(self, trajetorias, massas, tamanhoJanela, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.trajetorias = [normalizar(traj, media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]

        self.indices = []  # (idxTrajetoria, posicaoInicial)
        for idxTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - tamanhoJanela  # -1 posição sobra pro alvo
            for i in range(max(nAmostras, 0)):
                self.indices.append((idxTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idxTraj, i = self.indices[idx]
        traj = self.trajetorias[idxTraj]
        janela = traj[i:i + self.tamanhoJanela]      # (K, 18)
        alvo = traj[i + self.tamanhoJanela]           # (18,)
        massa = self.massas[idxTraj]                  # (3,)
        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massa, dtype=t.float32),
                t.tensor(alvo, dtype=t.float32))


# ---------------------------------------------------------------
# 4. MODELO: MLP simples
# ---------------------------------------------------------------
class MLPMinima(t.nn.Module):
    
    def __init__(self, tamanhoJanela, dimEstado=18, dimMassa=3, hidden=128):
        super().__init__()
        entrada = tamanhoJanela * dimEstado + dimMassa
        '''self.rede = t.nn.Sequential(
            t.nn.Linear(entrada, hidden),
            t.nn.ReLU(),
            t.nn.Linear(hidden, hidden),
            t.nn.ReLU(),
            t.nn.Linear(hidden, dimEstado),
        )'''
        self.rede = t.nn.Sequential(
            t.nn.Linear(entrada, hidden),
            t.nn.ReLU(),
            t.nn.Linear(hidden, dimEstado),
        )

    def forward(self, janela, massa):
        x = janela.flatten(start_dim=1)     # (B, K, 18) -> (B, K*18)
        x = t.cat([x, massa], dim=1)        # concatena massas
        return self.rede(x)                 # (B, 18)


# ---------------------------------------------------------------
# 5. TREINO + AVALIAÇÃO SIMPLES
# ---------------------------------------------------------------
if __name__ == "__main__":
    TAMANHO_JANELA = 10
    N_EPOCAS = 5
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    trajetorias, massas = carregarTrajetorias("simulacoesArtificiais/simulacoes3C")

    nValidacao = max(1, len(trajetorias) // 5)
    trajetoriasTreino, trajetoriasValidacao = trajetorias[nValidacao:], trajetorias[:nValidacao]
    massasTreino, massasValidacao = massas[nValidacao:], massas[:nValidacao]

    # normalização calculada SOMENTE com dados de treino
    media, desvio = calcularNormalizacao(trajetoriasTreino)
    mediaMassa = np.mean(massasTreino, axis=0)
    desvioMassa = np.std(massasTreino, axis=0)
    desvioMassa = np.where(desvioMassa < 1e-8, 1.0, desvioMassa)

    datasetTreino = JanelaProximoEstado(trajetoriasTreino, massasTreino, TAMANHO_JANELA,
                                         media, desvio, mediaMassa, desvioMassa)
    datasetValidacao = JanelaProximoEstado(trajetoriasValidacao, massasValidacao, TAMANHO_JANELA,
                                            media, desvio, mediaMassa, desvioMassa)

    loaderTreino = DataLoader(datasetTreino, batch_size=32, shuffle=True)
    loaderValidacao = DataLoader(datasetValidacao, batch_size=32, shuffle=False)

    modelo = MLPMinima(tamanhoJanela=TAMANHO_JANELA).to(device)
    otimizador = t.optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = t.nn.MSELoss()

    for epoca in range(1, N_EPOCAS + 1):
        modelo.train()
        lossTreino = 0.0
        for janela, massa, alvo in loaderTreino:
            janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)

            otimizador.zero_grad()
            predicao = modelo(janela, massa)
            loss = criterio(predicao, alvo)
            loss.backward()
            otimizador.step()

            lossTreino += loss.item()
        lossTreino /= len(loaderTreino)

        modelo.eval()
        lossVal = 0.0
        with t.no_grad():
            for janela, massa, alvo in loaderValidacao:
                janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)
                predicao = modelo(janela, massa)
                lossVal += criterio(predicao, alvo).item()
        lossVal /= len(loaderValidacao)

        print(f"Época {epoca:3d} | treino: {lossTreino:.6f} | val: {lossVal:.6f}")

    t.save(modelo.state_dict(), "mlp_minima.pt")

    # -----------------------------------------------------------
    # Rollout autoregressivo simples numa trajetória de teste:
    # a cada passo, a predição da rede vira parte da janela de
    # entrada do passo seguinte (é assim que se usa o modelo "de
    # verdade", já que na prática não se tem o futuro real).
    # -----------------------------------------------------------
    def rollout(modelo, trajetoria, massa, tamanhoJanela, nPassos, device):
        modelo.eval()
        trajNorm = normalizar(trajetoria, media, desvio)
        massaNorm = normalizar(massa, mediaMassa, desvioMassa)
        massaTensor = t.tensor(massaNorm, dtype=t.float32).unsqueeze(0).to(device)

        janelaAtual = trajNorm[:tamanhoJanela].copy()
        previsoes = []
        with t.no_grad():
            for _ in range(nPassos):
                janelaTensor = t.tensor(janelaAtual, dtype=t.float32).unsqueeze(0).to(device)
                pred = modelo(janelaTensor, massaTensor).squeeze(0).cpu().numpy()
                previsoes.append(pred)
                janelaAtual = np.vstack([janelaAtual[1:], pred])
        return desnormalizar(np.array(previsoes), media, desvio)

    trajTeste = trajetoriasValidacao[0]
    massaTeste = massasValidacao[0]
    nPassos = min(300, len(trajTeste) - TAMANHO_JANELA)

    previsoes = rollout(modelo, trajTeste, massaTeste, TAMANHO_JANELA, nPassos, device)
    alvos = trajTeste[TAMANHO_JANELA:TAMANHO_JANELA + nPassos]

    erro = np.linalg.norm(previsoes - alvos, axis=1)
    print(f"\nErro médio (norma L2 do vetor de 18 dim) no rollout: {erro.mean():.4f}")

    # plot simples: posição X do corpo 1, real vs. predito
    plt.figure(figsize=(10, 5))
    plt.plot(alvos[:, 0], label="Real (corpo 1, X)")
    plt.plot(previsoes[:, 0], "--", label="Predito (corpo 1, X)")
    plt.xlabel("Passo")
    plt.ylabel("Posição X")
    plt.title("Rollout autoregressivo — MLP mínima")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()"""

"""import numpy as np
import torch as t
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =====================================================================
# EXEMPLO MÍNIMO: MLP que prevê o PRÓXIMO ESTADO do problema de 3 corpos
# =====================================================================
# Objetivo: mostrar a estrutura MÍNIMA necessária para uma rede neural
# aprender a prever o próximo estado (posição+velocidade dos 3 corpos)
# a partir de uma janela de estados passados.
#
# O que foi DELIBERADAMENTE OMITIDO em relação ao script completo
# (isso é refinamento, não requisito):
#   - data augmentation por rotação
#   - scheduled sampling / rollout multi-passo no treino
#   - dropout / weight_decay
#   - métricas físicas (momento linear/angular, horizonte de previsão)
#   - previsão do Hamiltoniano (fica pra depois, como você pediu)
#
# Estrutura mínima = 5 blocos, e é só isso que uma rede precisa pra
# "funcionar" nesse problema:
#   1. Carregar dados (trajetórias simuladas por um integrador)
#   2. Normalizar (z-score) — sem isso o treino não converge bem,
#      pois posição e velocidade têm escalas muito diferentes
#   3. Dataset: janela de K passos passados -> 1 passo futuro
#   4. Modelo: MLP simples (poucas camadas densas + ReLU)
#   5. Loop de treino: forward -> loss (MSE) -> backward -> step
#
# NOVO NESTA VERSÃO: em vez de rodar o rollout autoregressivo em uma
# única trajetória de validação, roda em TODAS elas e desenha tudo
# numa ÚNICA figura (grade de subplots) — isso permite diferenciar
# um viés sistemático do modelo (aparece em quase todas as
# trajetórias, com inclinação parecida) de um problema específico de
# uma trajetória isolada (aparece só em uma ou poucas).
# =====================================================================


# ---------------------------------------------------------------
# 1. CARREGAR DADOS
# ---------------------------------------------------------------
def carregarTrajetorias(pastaNPZ):
    
    trajetorias, massasLista = [], []
    for arquivo in sorted(Path(pastaNPZ).glob("*.npz")):
        dados = np.load(arquivo)
        estado = dados["estado"][:, :18]
        if estado.shape[0] == 0:
            continue
        trajetorias.append(estado)
        massasLista.append(dados["massas"])
    return trajetorias, massasLista


# ---------------------------------------------------------------
# 2. NORMALIZAÇÃO (z-score simples, coluna a coluna)
# ---------------------------------------------------------------
def calcularNormalizacao(trajetorias):
    todos = np.concatenate(trajetorias)          # (N, 18)
    media = todos.mean(axis=0)
    desvio = todos.std(axis=0)
    desvio[desvio < 1e-8] = 1.0
    return media, desvio


def normalizar(x, media, desvio):
    return (x - media) / desvio


def desnormalizar(x, media, desvio):
    return x * desvio + media


# ---------------------------------------------------------------
# 3. DATASET: janela de K passos -> próximo passo
# ---------------------------------------------------------------
class JanelaProximoEstado(Dataset):
    
    def __init__(self, trajetorias, massas, tamanhoJanela, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.trajetorias = [normalizar(traj, media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]

        self.indices = []  # (idxTrajetoria, posicaoInicial)
        for idxTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - tamanhoJanela  # -1 posição sobra pro alvo
            for i in range(max(nAmostras, 0)):
                self.indices.append((idxTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idxTraj, i = self.indices[idx]
        traj = self.trajetorias[idxTraj]
        janela = traj[i:i + self.tamanhoJanela]      # (K, 18)
        alvo = traj[i + self.tamanhoJanela]           # (18,)
        massa = self.massas[idxTraj]                  # (3,)
        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massa, dtype=t.float32),
                t.tensor(alvo, dtype=t.float32))


# ---------------------------------------------------------------
# 4. MODELO: MLP simples (1 camada oculta, versão reduzida)
# ---------------------------------------------------------------
class MLPMinima(t.nn.Module):
    
    def __init__(self, tamanhoJanela, dimEstado=18, dimMassa=3, hidden=128):
        super().__init__()
        entrada = tamanhoJanela * dimEstado + dimMassa
        self.rede = t.nn.Sequential(
            t.nn.Linear(entrada, hidden),
            t.nn.ReLU(),
            t.nn.Linear(hidden, dimEstado),
        )

    def forward(self, janela, massa):
        x = janela.flatten(start_dim=1)     # (B, K, 18) -> (B, K*18)
        x = t.cat([x, massa], dim=1)        # concatena massas
        return self.rede(x)                 # (B, 18)


# ---------------------------------------------------------------
# 5. TREINO + AVALIAÇÃO EM MÚLTIPLAS TRAJETÓRIAS
# ---------------------------------------------------------------
if __name__ == "__main__":
    TAMANHO_JANELA = 10
    N_EPOCAS = 5
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    trajetorias, massas = carregarTrajetorias("simulacoesArtificiais/simulacoes3C")

    nValidacao = max(1, len(trajetorias) // 5)
    trajetoriasTreino, trajetoriasValidacao = trajetorias[nValidacao:], trajetorias[:nValidacao]
    massasTreino, massasValidacao = massas[nValidacao:], massas[:nValidacao]

    # normalização calculada SOMENTE com dados de treino
    media, desvio = calcularNormalizacao(trajetoriasTreino)
    mediaMassa = np.mean(massasTreino, axis=0)
    desvioMassa = np.std(massasTreino, axis=0)
    desvioMassa = np.where(desvioMassa < 1e-8, 1.0, desvioMassa)

    datasetTreino = JanelaProximoEstado(trajetoriasTreino, massasTreino, TAMANHO_JANELA,
                                         media, desvio, mediaMassa, desvioMassa)
    datasetValidacao = JanelaProximoEstado(trajetoriasValidacao, massasValidacao, TAMANHO_JANELA,
                                            media, desvio, mediaMassa, desvioMassa)

    loaderTreino = DataLoader(datasetTreino, batch_size=32, shuffle=True)
    loaderValidacao = DataLoader(datasetValidacao, batch_size=32, shuffle=False)

    modelo = MLPMinima(tamanhoJanela=TAMANHO_JANELA).to(device)
    otimizador = t.optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = t.nn.MSELoss()

    for epoca in range(1, N_EPOCAS + 1):
        modelo.train()
        lossTreino = 0.0
        for janela, massa, alvo in loaderTreino:
            janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)

            otimizador.zero_grad()
            predicao = modelo(janela, massa)
            loss = criterio(predicao, alvo)
            loss.backward()
            otimizador.step()

            lossTreino += loss.item()
        lossTreino /= len(loaderTreino)

        modelo.eval()
        lossVal = 0.0
        with t.no_grad():
            for janela, massa, alvo in loaderValidacao:
                janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)
                predicao = modelo(janela, massa)
                lossVal += criterio(predicao, alvo).item()
        lossVal /= len(loaderValidacao)

        print(f"Época {epoca:3d} | treino: {lossTreino:.6f} | val: {lossVal:.6f}")

    t.save(modelo.state_dict(), "mlp_minima.pt")

    # -----------------------------------------------------------
    # Rollout autoregressivo simples numa trajetória de teste:
    # a cada passo, a predição da rede vira parte da janela de
    # entrada do passo seguinte (é assim que se usa o modelo "de
    # verdade", já que na prática não se tem o futuro real).
    # -----------------------------------------------------------
    def rollout(modelo, trajetoria, massa, tamanhoJanela, nPassos, device):
        modelo.eval()
        trajNorm = normalizar(trajetoria, media, desvio)
        massaNorm = normalizar(massa, mediaMassa, desvioMassa)
        massaTensor = t.tensor(massaNorm, dtype=t.float32).unsqueeze(0).to(device)

        janelaAtual = trajNorm[:tamanhoJanela].copy()
        previsoes = []
        with t.no_grad():
            for _ in range(nPassos):
                janelaTensor = t.tensor(janelaAtual, dtype=t.float32).unsqueeze(0).to(device)
                pred = modelo(janelaTensor, massaTensor).squeeze(0).cpu().numpy()
                previsoes.append(pred)
                janelaAtual = np.vstack([janelaAtual[1:], pred])
        return desnormalizar(np.array(previsoes), media, desvio)

    # -----------------------------------------------------------
    # Roda o rollout em TODAS as trajetórias de validação e plota
    # tudo numa única figura (grade de subplots). Isso permite
    # distinguir viés sistemático do modelo (aparece parecido em
    # quase todas as trajetórias) de um problema pontual de uma
    # trajetória específica (aparece só em uma ou poucas).
    # -----------------------------------------------------------
    nTrajetoriasTeste = len(trajetoriasValidacao)
    nColunas = min(3, nTrajetoriasTeste)
    nLinhas = int(np.ceil(nTrajetoriasTeste / nColunas))

    fig, eixos = plt.subplots(nLinhas, nColunas, figsize=(5 * nColunas, 4 * nLinhas), squeeze=False)

    errosMedios = []
    for idx in range(nTrajetoriasTeste):
        trajTeste = trajetoriasValidacao[idx]
        massaTeste = massasValidacao[idx]
        nPassos = min(300, len(trajTeste) - TAMANHO_JANELA)

        linha, coluna = divmod(idx, nColunas)
        ax = eixos[linha][coluna]

        if nPassos <= 0:
            ax.set_visible(False)
            continue

        previsoes = rollout(modelo, trajTeste, massaTeste, TAMANHO_JANELA, nPassos, device)
        alvos = trajTeste[TAMANHO_JANELA:TAMANHO_JANELA + nPassos]

        erro = np.linalg.norm(previsoes - alvos, axis=1)
        errosMedios.append(erro.mean())

        ax.plot(alvos[:, 0], label="Real (corpo 1, X)")
        ax.plot(previsoes[:, 0], "--", label="Predito (corpo 1, X)")
        ax.set_title(f"Trajetória {idx} | erro médio: {erro.mean():.3f}", fontsize=10)
        ax.set_xlabel("Passo")
        ax.set_ylabel("Posição X")
        ax.grid()
        ax.legend(fontsize=8)

    # apaga eixos sobrando, se o número de trajetórias não preencher a grade
    for idx in range(nTrajetoriasTeste, nLinhas * nColunas):
        linha, coluna = divmod(idx, nColunas)
        eixos[linha][coluna].set_visible(False)

    fig.suptitle("Rollout autoregressivo em todas as trajetórias de validação — MLP mínima")
    plt.tight_layout()
    plt.show()

    print(f"\nErro médio entre trajetórias : {np.mean(errosMedios):.4f}")
    print(f"Desvio padrão entre trajetórias: {np.std(errosMedios):.4f}")
    print(f"Erro mínimo / máximo           : {np.min(errosMedios):.4f} / {np.max(errosMedios):.4f}")"""
"""
import numpy as np
import torch as t
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =====================================================================
# MLP MÍNIMA + SCHEDULED SAMPLING (probabilidade FIXA, sem decaimento)
# =====================================================================
# Ponto de partida: a versão mínima anterior, que só usava teacher
# forcing puro (janela de treino sempre 100% real). O diagnóstico do
# rollout em múltiplas trajetórias mostrou um viés sistemático — quase
# toda trajetória predita "sobe" na mesma direção, independente da
# física real, sinal de que a rede nunca tinha visto suas PRÓPRIAS
# predições como entrada durante o treino.
#
# O QUE MUDOU EM RELAÇÃO À VERSÃO ANTERIOR (só isso, resto idêntico):
#   1) O Dataset agora devolve, além da janela de contexto, um
#      HORIZONTE de H passos futuros reais (antes só devolvia 1).
#   2) O loop de treino "desenrola" H passos por amostra: a cada
#      passo do desenrolamento, decide (com probabilidade P_TEACHER_FORCING
#      FIXA, igual em toda época) se alimenta o próximo passo da janela
#      com o dado real ou com a própria predição do modelo.
#   3) A loss de treino agora é a média das losses dos H passos do
#      desenrolamento (antes era só a loss do passo único).
#
# P_TEACHER_FORCING é FIXO (não decai com a época, ao contrário do
# script completo original) — isso isola o efeito do scheduled
# sampling em si, sem misturar com o efeito do decaimento.
#
# Validação e avaliação (rollout em múltiplas trajetórias, plot em
# grade) continuam iguais à versão anterior, para comparação direta
# do mesmo gráfico antes/depois.
# =====================================================================


# ---------------------------------------------------------------
# 1. CARREGAR DADOS
# ---------------------------------------------------------------
def carregarTrajetorias(pastaNPZ):
    trajetorias, massasLista = [], []
    for arquivo in sorted(Path(pastaNPZ).glob("*.npz")):
        dados = np.load(arquivo)
        estado = dados["estado"]
        if estado.ndim < 2 or estado.shape[0] == 0:
            print(f"[aviso] pulando {arquivo.name}: trajetória vazia")
            continue
        estado = estado[:, :18]
        trajetorias.append(estado)
        massasLista.append(dados["massas"])
    return trajetorias, massasLista


# ---------------------------------------------------------------
# 2. NORMALIZAÇÃO (z-score simples, coluna a coluna)
# ---------------------------------------------------------------
def calcularNormalizacao(trajetorias):
    todos = np.concatenate(trajetorias)          # (N, 18)
    media = todos.mean(axis=0)
    desvio = todos.std(axis=0)
    desvio[desvio < 1e-8] = 1.0
    return media, desvio


def normalizar(x, media, desvio):
    return (x - media) / desvio


def desnormalizar(x, media, desvio):
    return x * desvio + media


# ---------------------------------------------------------------
# 3a. DATASET DE TREINO: janela de K passos -> HORIZONTE de H passos
# ---------------------------------------------------------------
class JanelaComHorizonte(Dataset):
    
    def __init__(self, trajetorias, massas, tamanhoJanela, horizonte, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.horizonte = horizonte
        self.trajetorias = [normalizar(traj, media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]

        comprimentoTotal = tamanhoJanela + horizonte
        self.indices = []  # (idxTrajetoria, posicaoInicial)
        for idxTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - comprimentoTotal + 1
            for i in range(max(nAmostras, 0)):
                self.indices.append((idxTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idxTraj, i = self.indices[idx]
        traj = self.trajetorias[idxTraj]
        janela = traj[i:i + self.tamanhoJanela]                              # (K, 18)
        futuros = traj[i + self.tamanhoJanela:i + self.tamanhoJanela + self.horizonte]  # (H, 18)
        massa = self.massas[idxTraj]                                         # (3,)
        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massa, dtype=t.float32),
                t.tensor(futuros, dtype=t.float32))


# ---------------------------------------------------------------
# 3b. DATASET DE VALIDAÇÃO: igual ao original (janela -> 1 passo)
# ---------------------------------------------------------------
class JanelaProximoEstado(Dataset):
    
    def __init__(self, trajetorias, massas, tamanhoJanela, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.trajetorias = [normalizar(traj, media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]

        self.indices = []
        for idxTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - tamanhoJanela
            for i in range(max(nAmostras, 0)):
                self.indices.append((idxTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idxTraj, i = self.indices[idx]
        traj = self.trajetorias[idxTraj]
        janela = traj[i:i + self.tamanhoJanela]
        alvo = traj[i + self.tamanhoJanela]
        massa = self.massas[idxTraj]
        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massa, dtype=t.float32),
                t.tensor(alvo, dtype=t.float32))


# ---------------------------------------------------------------
# 4. MODELO: MLP simples (1 camada oculta, igual à versão anterior)
# ---------------------------------------------------------------
class MLPMinima(t.nn.Module):
    def __init__(self, tamanhoJanela, dimEstado=18, dimMassa=3, hidden=128):
        super().__init__()
        entrada = tamanhoJanela * dimEstado + dimMassa
        self.rede = t.nn.Sequential(
            t.nn.Linear(entrada, hidden),
            t.nn.ReLU(),
            t.nn.Linear(hidden, dimEstado),
        )

    def forward(self, janela, massa):
        x = janela.flatten(start_dim=1)     # (B, K, 18) -> (B, K*18)
        x = t.cat([x, massa], dim=1)        # concatena massas
        return self.rede(x)                 # (B, 18)


# ---------------------------------------------------------------
# 5. TREINO COM SCHEDULED SAMPLING (probabilidade fixa) + AVALIAÇÃO
# ---------------------------------------------------------------
if __name__ == "__main__":
    TAMANHO_JANELA = 10
    HORIZONTE_TREINO = 5       # quantos passos desenrolar por amostra no treino
    P_TEACHER_FORCING = 0.5    # FIXO: 50% de chance de usar o dado real a cada passo do desenrolamento
    N_EPOCAS = 5
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    trajetorias, massas = carregarTrajetorias("simulacoesArtificiais/simulacoes3C")

    nValidacao = max(1, len(trajetorias) // 5)
    trajetoriasTreino, trajetoriasValidacao = trajetorias[nValidacao:], trajetorias[:nValidacao]
    massasTreino, massasValidacao = massas[nValidacao:], massas[:nValidacao]

    media, desvio = calcularNormalizacao(trajetoriasTreino)
    mediaMassa = np.mean(massasTreino, axis=0)
    desvioMassa = np.std(massasTreino, axis=0)
    desvioMassa = np.where(desvioMassa < 1e-8, 1.0, desvioMassa)

    datasetTreino = JanelaComHorizonte(trajetoriasTreino, massasTreino, TAMANHO_JANELA, HORIZONTE_TREINO,
                                        media, desvio, mediaMassa, desvioMassa)
    datasetValidacao = JanelaProximoEstado(trajetoriasValidacao, massasValidacao, TAMANHO_JANELA,
                                            media, desvio, mediaMassa, desvioMassa)

    loaderTreino = DataLoader(datasetTreino, batch_size=32, shuffle=True)
    loaderValidacao = DataLoader(datasetValidacao, batch_size=32, shuffle=False)

    modelo = MLPMinima(tamanhoJanela=TAMANHO_JANELA).to(device)
    otimizador = t.optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = t.nn.MSELoss()

    for epoca in range(1, N_EPOCAS + 1):
        modelo.train()
        lossTreino = 0.0
        for janela, massa, futuros in loaderTreino:
            janela = janela.to(device)      # (B, K, 18)
            massa = massa.to(device)        # (B, 3)
            futuros = futuros.to(device)    # (B, H, 18)

            otimizador.zero_grad()

            janelaAtual = janela
            lossPassos = []
            for h in range(HORIZONTE_TREINO):
                alvo = futuros[:, h, :]
                pred = modelo(janelaAtual, massa)
                lossPassos.append(criterio(pred, alvo))

                # scheduled sampling: por amostra do batch, decide se o
                # próximo passo da janela usa o dado real ou a própria
                # predição — com probabilidade P_TEACHER_FORCING FIXA
                usarReal = (t.rand(janelaAtual.size(0), 1, device=device) < P_TEACHER_FORCING).float()
                proximoEstado = usarReal * alvo + (1 - usarReal) * pred.detach()

                janelaAtual = t.cat([janelaAtual[:, 1:, :], proximoEstado.unsqueeze(1)], dim=1)

            loss = t.stack(lossPassos).mean()
            loss.backward()
            otimizador.step()

            lossTreino += loss.item()
        lossTreino /= len(loaderTreino)

        modelo.eval()
        lossVal = 0.0
        with t.no_grad():
            for janela, massa, alvo in loaderValidacao:
                janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)
                predicao = modelo(janela, massa)
                lossVal += criterio(predicao, alvo).item()
        lossVal /= len(loaderValidacao)

        print(f"Época {epoca:3d} | treino: {lossTreino:.6f} | val: {lossVal:.6f}")

    t.save(modelo.state_dict(), "mlp_scheduled_sampling.pt")

    # -----------------------------------------------------------
    # Rollout autoregressivo (idêntico à versão anterior)
    # -----------------------------------------------------------
    def rollout(modelo, trajetoria, massa, tamanhoJanela, nPassos, device):
        modelo.eval()
        trajNorm = normalizar(trajetoria, media, desvio)
        massaNorm = normalizar(massa, mediaMassa, desvioMassa)
        massaTensor = t.tensor(massaNorm, dtype=t.float32).unsqueeze(0).to(device)

        janelaAtual = trajNorm[:tamanhoJanela].copy()
        previsoes = []
        with t.no_grad():
            for _ in range(nPassos):
                janelaTensor = t.tensor(janelaAtual, dtype=t.float32).unsqueeze(0).to(device)
                pred = modelo(janelaTensor, massaTensor).squeeze(0).cpu().numpy()
                previsoes.append(pred)
                janelaAtual = np.vstack([janelaAtual[1:], pred])
        return desnormalizar(np.array(previsoes), media, desvio)

    # -----------------------------------------------------------
    # Rollout em TODAS as trajetórias de validação, numa única
    # figura em grade — igual à versão anterior, para comparação
    # direta do mesmo gráfico antes/depois do scheduled sampling.
    # -----------------------------------------------------------
    nTrajetoriasTeste = len(trajetoriasValidacao)
    nColunas = min(3, nTrajetoriasTeste)
    nLinhas = int(np.ceil(nTrajetoriasTeste / nColunas))

    fig, eixos = plt.subplots(nLinhas, nColunas, figsize=(5 * nColunas, 4 * nLinhas), squeeze=False)

    errosMedios = []
    for idx in range(nTrajetoriasTeste):
        trajTeste = trajetoriasValidacao[idx]
        massaTeste = massasValidacao[idx]
        nPassos = min(300, len(trajTeste) - TAMANHO_JANELA)

        linha, coluna = divmod(idx, nColunas)
        ax = eixos[linha][coluna]

        if nPassos <= 0:
            ax.set_visible(False)
            continue

        previsoes = rollout(modelo, trajTeste, massaTeste, TAMANHO_JANELA, nPassos, device)
        alvos = trajTeste[TAMANHO_JANELA:TAMANHO_JANELA + nPassos]

        erro = np.linalg.norm(previsoes - alvos, axis=1)
        errosMedios.append(erro.mean())

        ax.plot(alvos[:, 0], label="Real (corpo 1, X)")
        ax.plot(previsoes[:, 0], "--", label="Predito (corpo 1, X)")
        ax.set_title(f"Trajetória {idx} | erro médio: {erro.mean():.3f}", fontsize=10)
        ax.set_xlabel("Passo")
        ax.set_ylabel("Posição X")
        ax.grid()
        ax.legend(fontsize=8)

    for idx in range(nTrajetoriasTeste, nLinhas * nColunas):
        linha, coluna = divmod(idx, nColunas)
        eixos[linha][coluna].set_visible(False)

    fig.suptitle(f"Rollout com scheduled sampling (p={P_TEACHER_FORCING} fixo) — todas trajetórias de validação")
    plt.tight_layout()
    plt.show()

    print(f"\nErro médio entre trajetórias : {np.mean(errosMedios):.4f}")
    print(f"Desvio padrão entre trajetórias: {np.std(errosMedios):.4f}")
    print(f"Erro mínimo / máximo           : {np.min(errosMedios):.4f} / {np.max(errosMedios):.4f}")"""

"""import numpy as np
import torch as t
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =====================================================================
# GRU RESIDUAL + CURRÍCULO DE HORIZONTE
# =====================================================================
# Este script resolve o MESMO problema do script "MLP mínima +
# scheduled sampling" (prever o próximo estado de um sistema de 3
# corpos), mas com escolhas de projeto DIFERENTES em três pontos-chave.
# O carregamento de dados, a normalização e a avaliação por rollout em
# grade foram mantidos iguais de propósito, para que os dois scripts
# fiquem comparáveis lado a lado (mesmo dado, mesma métrica).
#
# DIFERENÇA 1 — ARQUITETURA: recorrência em vez de janela achatada
#   A versão MLP transforma a janela de K passos num vetor único
#   (K*18) e joga numa rede feed-forward. Aqui, a janela é tratada
#   como uma SEQUÊNCIA: uma GRU lê os K passos em ordem e produz um
#   resumo (hidden state) do histórico recente do sistema. Isso deixa
#   explícito para a rede que existe uma ordem temporal entre os
#   passos — a MLP não sabe disso, só vê números concatenados.
#
# DIFERENÇA 2 — SAÍDA: delta (resíduo) em vez de estado absoluto
#   A versão MLP prediz diretamente o próximo estado. Aqui a rede
#   prediz a VARIAÇÃO entre o último estado da janela e o próximo
#   passo; o estado previsto é ultimoEstado + delta (conexão
#   residual). Para grandezas físicas que mudam pouco de um passo
#   para o outro (posição, velocidade em dt pequeno), aprender "quanto
#   muda" costuma ser mais fácil e estável do que aprender "qual é o
#   valor absoluto".
#
# DIFERENÇA 3 — TREINO MULTI-PASSO: currículo de horizonte crescente
#   em vez de scheduled sampling com probabilidade fixa
#   A versão MLP, a cada passo do desenrolamento, sorteia se usa o
#   dado real ou a própria predição (probabilidade fixa de 50%). Aqui
#   a estratégia é outra: o desenrolamento é SEMPRE 100% autoregressivo
#   (a rede só vê suas próprias predições, nunca o dado real durante o
#   desenrolamento), mas o TAMANHO do horizonte cresce ao longo das
#   épocas — começa em 1 passo (quase teacher forcing puro) e vai até
#   HORIZONTE_MAX. A ideia: nas primeiras épocas o modelo ainda erra
#   muito, então pedir horizontes longos autoregressivos só ensina a
#   propagar erro; conforme o modelo melhora, aumentar o horizonte
#   força a rede a lidar com o próprio erro acumulado aos poucos.
#
# Ambas as estratégias (scheduled sampling e currículo de horizonte)
# atacam o mesmo problema — exposure bias / discrepância entre treino
# (teacher forcing) e uso real (autoregressivo) — só que por caminhos
# diferentes. Comparar as duas é o ponto deste exercício.
# =====================================================================


# ---------------------------------------------------------------
# 1. CARREGAR DADOS (idêntico ao script anterior — mesmo formato de
#    arquivo .npz, então não há motivo para mudar aqui)
# ---------------------------------------------------------------
def carregarTrajetorias(pastaNPZ):
    trajetorias, massasLista = [], []
    for arquivo in sorted(Path(pastaNPZ).glob("*.npz")):
        dados = np.load(arquivo)
        estado = dados["estado"]
        if estado.ndim < 2 or estado.shape[0] == 0:
            print(f"[aviso] pulando {arquivo.name}: trajetória vazia")
            continue
        estado = estado[:, :18]
        trajetorias.append(estado)
        massasLista.append(dados["massas"])
    return trajetorias, massasLista


# ---------------------------------------------------------------
# 2. NORMALIZAÇÃO (idêntica — z-score coluna a coluna)
# ---------------------------------------------------------------
def calcularNormalizacao(trajetorias):
    todos = np.concatenate(trajetorias)
    media = todos.mean(axis=0)
    desvio = todos.std(axis=0)
    desvio[desvio < 1e-8] = 1.0
    return media, desvio


def normalizar(x, media, desvio):
    return (x - media) / desvio


def desnormalizar(x, media, desvio):
    return x * desvio + media


# ---------------------------------------------------------------
# 3a. DATASET DE TREINO: janela de K passos -> HORIZONTE_MAX passos
#     futuros reais (o currículo escolhe, no loop de treino, quantos
#     desses passos usar em cada época)
# ---------------------------------------------------------------
class SequenciaComHorizonte(Dataset):
    def __init__(self, trajetorias, massas, tamanhoJanela, horizonteMax, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.horizonteMax = horizonteMax
        self.trajetorias = [normalizar(traj, media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]

        comprimentoTotal = tamanhoJanela + horizonteMax
        self.indices = []
        for idxTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - comprimentoTotal + 1
            for i in range(max(nAmostras, 0)):
                self.indices.append((idxTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idxTraj, i = self.indices[idx]
        traj = self.trajetorias[idxTraj]
        janela = traj[i:i + self.tamanhoJanela]
        futuros = traj[i + self.tamanhoJanela:i + self.tamanhoJanela + self.horizonteMax]
        massa = self.massas[idxTraj]
        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massa, dtype=t.float32),
                t.tensor(futuros, dtype=t.float32))


# ---------------------------------------------------------------
# 3b. DATASET DE VALIDAÇÃO: janela -> 1 passo (mesmo critério do
#     script anterior, para as duas versões serem comparáveis)
# ---------------------------------------------------------------
class SequenciaProximoEstado(Dataset):
    def __init__(self, trajetorias, massas, tamanhoJanela, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.trajetorias = [normalizar(traj, media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]

        self.indices = []
        for idxTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - tamanhoJanela
            for i in range(max(nAmostras, 0)):
                self.indices.append((idxTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idxTraj, i = self.indices[idx]
        traj = self.trajetorias[idxTraj]
        janela = traj[i:i + self.tamanhoJanela]
        alvo = traj[i + self.tamanhoJanela]
        massa = self.massas[idxTraj]
        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massa, dtype=t.float32),
                t.tensor(alvo, dtype=t.float32))


# ---------------------------------------------------------------
# 4. MODELO: GRU + cabeça residual (em vez de MLP feed-forward)
# ---------------------------------------------------------------
class GRUResidual(t.nn.Module):
    
    def __init__(self, dimEstado=18, dimMassa=3, hiddenGRU=64, hiddenCabeca=64, numCamadasGRU=1):
        super().__init__()
        self.gru = t.nn.GRU(input_size=dimEstado, hidden_size=hiddenGRU,
                             num_layers=numCamadasGRU, batch_first=True)
        self.cabeca = t.nn.Sequential(
            t.nn.Linear(hiddenGRU + dimMassa, hiddenCabeca),
            t.nn.ReLU(),
            t.nn.Linear(hiddenCabeca, dimEstado),
        )

    def forward(self, janela, massa):
        _, hFinal = self.gru(janela)        # hFinal: (numCamadasGRU, B, hiddenGRU)
        resumo = hFinal[-1]                 # (B, hiddenGRU) — última camada da GRU
        entradaCabeca = t.cat([resumo, massa], dim=1)
        delta = self.cabeca(entradaCabeca)  # (B, 18) — variação prevista
        ultimoEstado = janela[:, -1, :]     # (B, 18) — último estado real/predito da janela
        return ultimoEstado + delta         # conexão residual: estado = último + variação


# ---------------------------------------------------------------
# 5. TREINO COM CURRÍCULO DE HORIZONTE + AVALIAÇÃO
# ---------------------------------------------------------------
if __name__ == "__main__":
    TAMANHO_JANELA = 10
    HORIZONTE_MAX = 5      # horizonte final do currículo (mesmo valor do HORIZONTE_TREINO anterior)
    N_EPOCAS = 5
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    trajetorias, massas = carregarTrajetorias("simulacoesArtificiais/simulacoes3C")

    nValidacao = max(1, len(trajetorias) // 5)
    trajetoriasTreino, trajetoriasValidacao = trajetorias[nValidacao:], trajetorias[:nValidacao]
    massasTreino, massasValidacao = massas[nValidacao:], massas[:nValidacao]

    media, desvio = calcularNormalizacao(trajetoriasTreino)
    mediaMassa = np.mean(massasTreino, axis=0)
    desvioMassa = np.std(massasTreino, axis=0)
    desvioMassa = np.where(desvioMassa < 1e-8, 1.0, desvioMassa)

    datasetTreino = SequenciaComHorizonte(trajetoriasTreino, massasTreino, TAMANHO_JANELA, HORIZONTE_MAX,
                                           media, desvio, mediaMassa, desvioMassa)
    datasetValidacao = SequenciaProximoEstado(trajetoriasValidacao, massasValidacao, TAMANHO_JANELA,
                                               media, desvio, mediaMassa, desvioMassa)

    loaderTreino = DataLoader(datasetTreino, batch_size=32, shuffle=True)
    loaderValidacao = DataLoader(datasetValidacao, batch_size=32, shuffle=False)

    modelo = GRUResidual().to(device)
    otimizador = t.optim.Adam(modelo.parameters(), lr=1e-3)
    criterio = t.nn.MSELoss()

    for epoca in range(1, N_EPOCAS + 1):
        # Currículo: horizonte cresce linearmente de 1 até HORIZONTE_MAX
        # ao longo das épocas (na última época, horizonte = HORIZONTE_MAX).
        if N_EPOCAS > 1:
            horizonteAtual = 1 + (epoca - 1) * (HORIZONTE_MAX - 1) // (N_EPOCAS - 1)
        else:
            horizonteAtual = HORIZONTE_MAX

        modelo.train()
        lossTreino = 0.0
        for janela, massa, futuros in loaderTreino:
            janela = janela.to(device)      # (B, K, 18)
            massa = massa.to(device)        # (B, 3)
            futuros = futuros.to(device)    # (B, HORIZONTE_MAX, 18)

            otimizador.zero_grad()

            janelaAtual = janela
            lossPassos = []
            for h in range(horizonteAtual):
                alvo = futuros[:, h, :]
                pred = modelo(janelaAtual, massa)
                lossPassos.append(criterio(pred, alvo))

                # 100% autoregressivo dentro do desenrolamento: a
                # janela sempre avança com a PRÓPRIA predição, nunca
                # com o dado real (diferente do scheduled sampling).
                janelaAtual = t.cat([janelaAtual[:, 1:, :], pred.detach().unsqueeze(1)], dim=1)

            loss = t.stack(lossPassos).mean()
            loss.backward()
            otimizador.step()

            lossTreino += loss.item()
        lossTreino /= len(loaderTreino)

        modelo.eval()
        lossVal = 0.0
        with t.no_grad():
            for janela, massa, alvo in loaderValidacao:
                janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)
                predicao = modelo(janela, massa)
                lossVal += criterio(predicao, alvo).item()
        lossVal /= len(loaderValidacao)

        print(f"Época {epoca:3d} | horizonte: {horizonteAtual} | treino: {lossTreino:.6f} | val: {lossVal:.6f}")

    t.save(modelo.state_dict(), "gru_residual_curriculo.pt")

    # -----------------------------------------------------------
    # Rollout autoregressivo — a interface é idêntica ao script
    # anterior (janela -> próximo estado), então a função de rollout
    # e o gráfico em grade são reaproveitados sem nenhuma mudança.
    # A diferença toda está dentro de `modelo(...)`.
    # -----------------------------------------------------------
    def rollout(modelo, trajetoria, massa, tamanhoJanela, nPassos, device):
        modelo.eval()
        trajNorm = normalizar(trajetoria, media, desvio)
        massaNorm = normalizar(massa, mediaMassa, desvioMassa)
        massaTensor = t.tensor(massaNorm, dtype=t.float32).unsqueeze(0).to(device)

        janelaAtual = trajNorm[:tamanhoJanela].copy()
        previsoes = []
        with t.no_grad():
            for _ in range(nPassos):
                janelaTensor = t.tensor(janelaAtual, dtype=t.float32).unsqueeze(0).to(device)
                pred = modelo(janelaTensor, massaTensor).squeeze(0).cpu().numpy()
                previsoes.append(pred)
                janelaAtual = np.vstack([janelaAtual[1:], pred])
        return desnormalizar(np.array(previsoes), media, desvio)

    nTrajetoriasTeste = len(trajetoriasValidacao)
    nColunas = min(3, nTrajetoriasTeste)
    nLinhas = int(np.ceil(nTrajetoriasTeste / nColunas))

    fig, eixos = plt.subplots(nLinhas, nColunas, figsize=(5 * nColunas, 4 * nLinhas), squeeze=False)

    errosMedios = []
    for idx in range(nTrajetoriasTeste):
        trajTeste = trajetoriasValidacao[idx]
        massaTeste = massasValidacao[idx]
        nPassos = min(300, len(trajTeste) - TAMANHO_JANELA)

        linha, coluna = divmod(idx, nColunas)
        ax = eixos[linha][coluna]

        if nPassos <= 0:
            ax.set_visible(False)
            continue

        previsoes = rollout(modelo, trajTeste, massaTeste, TAMANHO_JANELA, nPassos, device)
        alvos = trajTeste[TAMANHO_JANELA:TAMANHO_JANELA + nPassos]

        erro = np.linalg.norm(previsoes - alvos, axis=1)
        errosMedios.append(erro.mean())

        ax.plot(alvos[:, 0], label="Real (corpo 1, X)")
        ax.plot(previsoes[:, 0], "--", label="Predito (corpo 1, X)")
        ax.set_title(f"Trajetória {idx} | erro médio: {erro.mean():.3f}", fontsize=10)
        ax.set_xlabel("Passo")
        ax.set_ylabel("Posição X")
        ax.grid()
        ax.legend(fontsize=8)

    for idx in range(nTrajetoriasTeste, nLinhas * nColunas):
        linha, coluna = divmod(idx, nColunas)
        eixos[linha][coluna].set_visible(False)

    fig.suptitle(f"Rollout com GRU residual + currículo de horizonte (até {HORIZONTE_MAX}) — validação")
    plt.tight_layout()
    plt.show()

    print(f"\nErro médio entre trajetórias : {np.mean(errosMedios):.4f}")
    print(f"Desvio padrão entre trajetórias: {np.std(errosMedios):.4f}")
    print(f"Erro mínimo / máximo           : {np.min(errosMedios):.4f} / {np.max(errosMedios):.4f}")"""

"""import numpy as np
import torch as t
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =====================================================================
# GRU (ESTADO ABSOLUTO) + CURRÍCULO DE HORIZONTE
# =====================================================================
# Este script resolve o MESMO problema do script "MLP mínima +
# scheduled sampling" (prever o próximo estado de um sistema de 3
# corpos), mas com escolhas de projeto DIFERENTES em dois pontos-chave.
# A rede AQUI PREDIZ O ESTADO ABSOLUTO diretamente (nada de delta ou
# conexão residual) — igual ao script original nesse aspecto. O
# carregamento de dados, a normalização e a avaliação por rollout em
# grade também foram mantidos iguais de propósito, para que os dois
# scripts fiquem comparáveis lado a lado (mesmo dado, mesma métrica).
#
# DIFERENÇA 1 — ARQUITETURA: recorrência em vez de janela achatada
#   A versão MLP transforma a janela de K passos num vetor único
#   (K*18) e joga numa rede feed-forward. Aqui, a janela é tratada
#   como uma SEQUÊNCIA: uma GRU lê os K passos em ordem e produz um
#   resumo (hidden state) do histórico recente do sistema; esse resumo
#   (mais as massas) alimenta uma cabeça que prediz o próximo estado
#   absoluto diretamente. Isso deixa explícito para a rede que existe
#   uma ordem temporal entre os passos — a MLP não sabe disso, só vê
#   números concatenados.
#
# DIFERENÇA 2 — TREINO MULTI-PASSO: currículo de horizonte crescente
#   em vez de scheduled sampling com probabilidade fixa
#   A versão MLP, a cada passo do desenrolamento, sorteia se usa o
#   dado real ou a própria predição (probabilidade fixa de 50%). Aqui
#   a estratégia é outra: o desenrolamento é SEMPRE 100% autoregressivo
#   (a rede só vê suas próprias predições, nunca o dado real durante o
#   desenrolamento), mas o TAMANHO do horizonte cresce ao longo das
#   épocas — começa em 1 passo (quase teacher forcing puro) e vai até
#   HORIZONTE_MAX. A ideia: nas primeiras épocas o modelo ainda erra
#   muito, então pedir horizontes longos autoregressivos só ensina a
#   propagar erro; conforme o modelo melhora, aumentar o horizonte
#   força a rede a lidar com o próprio erro acumulado aos poucos.
#
# Ambas as estratégias (scheduled sampling e currículo de horizonte)
# atacam o mesmo problema — exposure bias / discrepância entre treino
# (teacher forcing) e uso real (autoregressivo) — só que por caminhos
# diferentes. Comparar as duas é o ponto deste exercício.
# =====================================================================


# ---------------------------------------------------------------
# 1. CARREGAR DADOS (idêntico ao script anterior — mesmo formato de
#    arquivo .npz, então não há motivo para mudar aqui)
# ---------------------------------------------------------------
def carregarTrajetorias(pastaNPZ):
    trajetorias, massasLista = [], []
    for arquivo in sorted(Path(pastaNPZ).glob("*.npz")):
        dados = np.load(arquivo)
        estado = dados["estado"]
        if estado.ndim < 2 or estado.shape[0] == 0:
            print(f"[aviso] pulando {arquivo.name}: trajetória vazia")
            continue
        estado = estado[:, :18]
        trajetorias.append(estado)
        massasLista.append(dados["massas"])
    return trajetorias, massasLista


# ---------------------------------------------------------------
# 2. NORMALIZAÇÃO (idêntica — z-score coluna a coluna)
# ---------------------------------------------------------------
def calcularNormalizacao(trajetorias):
    todos = np.concatenate(trajetorias)
    media = todos.mean(axis=0)
    desvio = todos.std(axis=0)
    desvio[desvio < 1e-8] = 1.0
    return media, desvio


def normalizar(x, media, desvio):
    return (x - media) / desvio


def desnormalizar(x, media, desvio):
    return x * desvio + media


# ---------------------------------------------------------------
# 3a. DATASET DE TREINO: janela de K passos -> HORIZONTE_MAX passos
#     futuros reais (o currículo escolhe, no loop de treino, quantos
#     desses passos usar em cada época)
# ---------------------------------------------------------------
class SequenciaComHorizonte(Dataset):
    def __init__(self, trajetorias, massas, tamanhoJanela, horizonteMax, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.horizonteMax = horizonteMax
        self.trajetorias = [normalizar(traj, media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]

        comprimentoTotal = tamanhoJanela + horizonteMax
        self.indices = []
        for idxTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - comprimentoTotal + 1
            for i in range(max(nAmostras, 0)):
                self.indices.append((idxTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idxTraj, i = self.indices[idx]
        traj = self.trajetorias[idxTraj]
        janela = traj[i:i + self.tamanhoJanela]
        futuros = traj[i + self.tamanhoJanela:i + self.tamanhoJanela + self.horizonteMax]
        massa = self.massas[idxTraj]
        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massa, dtype=t.float32),
                t.tensor(futuros, dtype=t.float32))


# ---------------------------------------------------------------
# 3b. DATASET DE VALIDAÇÃO: janela -> 1 passo (mesmo critério do
#     script anterior, para as duas versões serem comparáveis)
# ---------------------------------------------------------------
class SequenciaProximoEstado(Dataset):
    def __init__(self, trajetorias, massas, tamanhoJanela, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.trajetorias = [normalizar(traj, media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]

        self.indices = []
        for idxTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - tamanhoJanela
            for i in range(max(nAmostras, 0)):
                self.indices.append((idxTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idxTraj, i = self.indices[idx]
        traj = self.trajetorias[idxTraj]
        janela = traj[i:i + self.tamanhoJanela]
        alvo = traj[i + self.tamanhoJanela]
        massa = self.massas[idxTraj]
        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massa, dtype=t.float32),
                t.tensor(alvo, dtype=t.float32))


# ---------------------------------------------------------------
# 4. MODELO: GRU + cabeça que prediz o estado absoluto (em vez de
#    MLP feed-forward sobre a janela achatada)
# ---------------------------------------------------------------
class GRURecorrente(t.nn.Module):
    
    def __init__(self, dimEstado=18, dimMassa=3, hiddenGRU=64, hiddenCabeca=64, numCamadasGRU=1):
        super().__init__()
        self.gru = t.nn.GRU(input_size=dimEstado, hidden_size=hiddenGRU,
                             num_layers=numCamadasGRU, batch_first=True)
        self.cabeca = t.nn.Sequential(
            t.nn.Linear(hiddenGRU + dimMassa, hiddenCabeca),
            t.nn.ReLU(),
            t.nn.Linear(hiddenCabeca, dimEstado),
        )

    def forward(self, janela, massa):
        _, hFinal = self.gru(janela)        # hFinal: (numCamadasGRU, B, hiddenGRU)
        resumo = hFinal[-1]                 # (B, hiddenGRU) — última camada da GRU
        entradaCabeca = t.cat([resumo, massa], dim=1)
        return self.cabeca(entradaCabeca)   # (B, 18) — próximo estado absoluto, predito direto


# ---------------------------------------------------------------
# 5. TREINO COM CURRÍCULO DE HORIZONTE + AVALIAÇÃO
# ---------------------------------------------------------------
if __name__ == "__main__":
    TAMANHO_JANELA = 10
    HORIZONTE_MAX = 5      # horizonte final do currículo (mesmo valor do HORIZONTE_TREINO anterior)
    N_EPOCAS = 5
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Se TREINAR_MODELO = False, o treino é pulado e os pesos salvos em
    # CAMINHO_MODELO_SALVO são carregados direto — útil para testar
    # vários arquivos .npz na seção 6 sem precisar retreinar toda vez.
    TREINAR_MODELO = True
    CAMINHO_MODELO_SALVO = "gru_curriculo_horizonte.pt"

    trajetorias, massas = carregarTrajetorias("simulacoesArtificiais/simulacoes3C")

    nValidacao = max(1, len(trajetorias) // 5)
    trajetoriasTreino, trajetoriasValidacao = trajetorias[nValidacao:], trajetorias[:nValidacao]
    massasTreino, massasValidacao = massas[nValidacao:], massas[:nValidacao]

    media, desvio = calcularNormalizacao(trajetoriasTreino)
    mediaMassa = np.mean(massasTreino, axis=0)
    desvioMassa = np.std(massasTreino, axis=0)
    desvioMassa = np.where(desvioMassa < 1e-8, 1.0, desvioMassa)

    datasetTreino = SequenciaComHorizonte(trajetoriasTreino, massasTreino, TAMANHO_JANELA, HORIZONTE_MAX,
                                           media, desvio, mediaMassa, desvioMassa)
    datasetValidacao = SequenciaProximoEstado(trajetoriasValidacao, massasValidacao, TAMANHO_JANELA,
                                               media, desvio, mediaMassa, desvioMassa)

    loaderTreino = DataLoader(datasetTreino, batch_size=32, shuffle=True)
    loaderValidacao = DataLoader(datasetValidacao, batch_size=32, shuffle=False)

    modelo = GRURecorrente().to(device)

    if TREINAR_MODELO:
        otimizador = t.optim.Adam(modelo.parameters(), lr=1e-3)
        criterio = t.nn.MSELoss()

        for epoca in range(1, N_EPOCAS + 1):
            # Currículo: horizonte cresce linearmente de 1 até HORIZONTE_MAX
            # ao longo das épocas (na última época, horizonte = HORIZONTE_MAX).
            if N_EPOCAS > 1:
                horizonteAtual = 1 + (epoca - 1) * (HORIZONTE_MAX - 1) // (N_EPOCAS - 1)
            else:
                horizonteAtual = HORIZONTE_MAX

            modelo.train()
            lossTreino = 0.0
            for janela, massa, futuros in loaderTreino:
                janela = janela.to(device)      # (B, K, 18)
                massa = massa.to(device)        # (B, 3)
                futuros = futuros.to(device)    # (B, HORIZONTE_MAX, 18)

                otimizador.zero_grad()

                janelaAtual = janela
                lossPassos = []
                for h in range(horizonteAtual):
                    alvo = futuros[:, h, :]
                    pred = modelo(janelaAtual, massa)
                    lossPassos.append(criterio(pred, alvo))

                    # 100% autoregressivo dentro do desenrolamento: a
                    # janela sempre avança com a PRÓPRIA predição, nunca
                    # com o dado real (diferente do scheduled sampling).
                    janelaAtual = t.cat([janelaAtual[:, 1:, :], pred.detach().unsqueeze(1)], dim=1)

                loss = t.stack(lossPassos).mean()
                loss.backward()
                otimizador.step()

                lossTreino += loss.item()
            lossTreino /= len(loaderTreino)

            modelo.eval()
            lossVal = 0.0
            with t.no_grad():
                for janela, massa, alvo in loaderValidacao:
                    janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)
                    predicao = modelo(janela, massa)
                    lossVal += criterio(predicao, alvo).item()
            lossVal /= len(loaderValidacao)

            print(f"Época {epoca:3d} | horizonte: {horizonteAtual} | treino: {lossTreino:.6f} | val: {lossVal:.6f}")

        t.save(modelo.state_dict(), CAMINHO_MODELO_SALVO)
    else:
        modelo.load_state_dict(t.load(CAMINHO_MODELO_SALVO, map_location=device))
        print(f"Treino pulado — pesos carregados de {CAMINHO_MODELO_SALVO}")

    # -----------------------------------------------------------
    # Rollout autoregressivo — a interface é idêntica ao script
    # anterior (janela -> próximo estado), então a função de rollout
    # e o gráfico em grade são reaproveitados sem nenhuma mudança.
    # A diferença toda está dentro de `modelo(...)`.
    # -----------------------------------------------------------
    def rollout(modelo, trajetoria, massa, tamanhoJanela, nPassos, device):
        modelo.eval()
        trajNorm = normalizar(trajetoria, media, desvio)
        massaNorm = normalizar(massa, mediaMassa, desvioMassa)
        massaTensor = t.tensor(massaNorm, dtype=t.float32).unsqueeze(0).to(device)

        janelaAtual = trajNorm[:tamanhoJanela].copy()
        previsoes = []
        with t.no_grad():
            for _ in range(nPassos):
                janelaTensor = t.tensor(janelaAtual, dtype=t.float32).unsqueeze(0).to(device)
                pred = modelo(janelaTensor, massaTensor).squeeze(0).cpu().numpy()
                previsoes.append(pred)
                janelaAtual = np.vstack([janelaAtual[1:], pred])
        return desnormalizar(np.array(previsoes), media, desvio)

    nTrajetoriasTeste = len(trajetoriasValidacao)
    nColunas = min(3, nTrajetoriasTeste)
    nLinhas = int(np.ceil(nTrajetoriasTeste / nColunas))

    fig, eixos = plt.subplots(nLinhas, nColunas, figsize=(5 * nColunas, 4 * nLinhas), squeeze=False)

    errosMedios = []
    for idx in range(nTrajetoriasTeste):
        trajTeste = trajetoriasValidacao[idx]
        massaTeste = massasValidacao[idx]
        nPassos = min(300, len(trajTeste) - TAMANHO_JANELA)

        linha, coluna = divmod(idx, nColunas)
        ax = eixos[linha][coluna]

        if nPassos <= 0:
            ax.set_visible(False)
            continue

        previsoes = rollout(modelo, trajTeste, massaTeste, TAMANHO_JANELA, nPassos, device)
        alvos = trajTeste[TAMANHO_JANELA:TAMANHO_JANELA + nPassos]

        erro = np.linalg.norm(previsoes - alvos, axis=1)
        errosMedios.append(erro.mean())

        ax.plot(alvos[:, 0], label="Real (corpo 1, X)")
        ax.plot(previsoes[:, 0], "--", label="Predito (corpo 1, X)")
        ax.set_title(f"Trajetória {idx} | erro médio: {erro.mean():.3f}", fontsize=10)
        ax.set_xlabel("Passo")
        ax.set_ylabel("Posição X")
        ax.grid()
        ax.legend(fontsize=8)

    for idx in range(nTrajetoriasTeste, nLinhas * nColunas):
        linha, coluna = divmod(idx, nColunas)
        eixos[linha][coluna].set_visible(False)

    fig.suptitle(f"Rollout com GRU residual + currículo de horizonte (até {HORIZONTE_MAX}) — validação")
    plt.tight_layout()
    plt.show()

    print(f"\nErro médio entre trajetórias : {np.mean(errosMedios):.4f}")
    print(f"Desvio padrão entre trajetórias: {np.std(errosMedios):.4f}")
    print(f"Erro mínimo / máximo           : {np.min(errosMedios):.4f} / {np.max(errosMedios):.4f}")

    # -----------------------------------------------------------
    # 6. TESTE EM UM ARQUIVO .npz ESPECÍFICO (real x previsto)
    # -----------------------------------------------------------
    # Edite ARQUIVO_TESTE para apontar para qualquer simulação — pode
    # ser um arquivo de treino, de validação, ou uma simulação nova que
    # nunca foi vista pela rede — e rode o script de novo (com
    # TREINAR_MODELO = False depois da primeira vez, para não perder
    # tempo retreinando) para ver a trajetória real sobreposta à
    # trajetória prevista pela rede, corpo a corpo.
    ARQUIVO_TESTE = "simulacoesArtificiais/simulacoes3C/simulacao3D_7.npz"  # None para pular esta seção
    NUM_CORPOS = 3  # assume estado = [corpo1: x,y,z,vx,vy,vz, corpo2: ..., corpo3: ...]

    def testarArquivoNPZ(modelo, caminhoNPZ, tamanhoJanela, device, nPassos=None):
        
        dados = np.load(caminhoNPZ)
        estado = dados["estado"][:, :18]
        massa = dados["massas"]

        if nPassos is None:
            nPassos = len(estado) - tamanhoJanela
        nPassos = min(nPassos, len(estado) - tamanhoJanela)
        if nPassos <= 0:
            raise ValueError(f"Trajetória de {caminhoNPZ} é curta demais para janela={tamanhoJanela}")

        previsto = rollout(modelo, estado, massa, tamanhoJanela, nPassos, device)
        real = estado[tamanhoJanela:tamanhoJanela + nPassos]
        return real, previsto

    def plotarComparacaoArquivo(real, previsto, nomeArquivo, numCorpos=3):
        
        fig, eixos = plt.subplots(numCorpos, 2, figsize=(10, 4 * numCorpos), squeeze=False)
        passos = np.arange(len(real))

        for corpo in range(numCorpos):
            col0 = corpo * 6
            xReal, yReal = real[:, col0], real[:, col0 + 1]
            xPrev, yPrev = previsto[:, col0], previsto[:, col0 + 1]
            erroPos = np.linalg.norm(real[:, col0:col0 + 3] - previsto[:, col0:col0 + 3], axis=1)

            axTraj = eixos[corpo][0]
            axTraj.plot(xReal, yReal, label="Real")
            axTraj.plot(xPrev, yPrev, "--", label="Previsto")
            axTraj.scatter([xReal[0]], [yReal[0]], color="green", zorder=5, label="Início")
            axTraj.set_title(f"Corpo {corpo + 1} — trajetória (plano XY)")
            axTraj.set_xlabel("X")
            axTraj.set_ylabel("Y")
            axTraj.axis("equal")
            axTraj.grid()
            axTraj.legend(fontsize=8)

            axErro = eixos[corpo][1]
            axErro.plot(passos, erroPos)
            axErro.set_title(f"Corpo {corpo + 1} — erro de posição (norma) ao longo do tempo")
            axErro.set_xlabel("Passo")
            axErro.set_ylabel("Erro")
            axErro.grid()

        fig.suptitle(f"Real x Previsto — {Path(nomeArquivo).name}")
        plt.tight_layout()
        plt.show()

    if ARQUIVO_TESTE is not None:
        real, previsto = testarArquivoNPZ(modelo, ARQUIVO_TESTE, TAMANHO_JANELA, device)
        plotarComparacaoArquivo(real, previsto, ARQUIVO_TESTE, numCorpos=NUM_CORPOS)

        erroGlobal = np.linalg.norm(real - previsto, axis=1)
        print(f"\nArquivo de teste: {ARQUIVO_TESTE}")
        print(f"Erro médio (todos os corpos): {erroGlobal.mean():.4f}")
        print(f"Erro no último passo simulado: {erroGlobal[-1]:.4f}")"""

import numpy as np
import torch as t
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =====================================================================
# GRU (ESTADO ABSOLUTO) + CURRÍCULO DE HORIZONTE
# =====================================================================
# Este script resolve o MESMO problema do script "MLP mínima +
# scheduled sampling" (prever o próximo estado de um sistema de 3
# corpos), mas com escolhas de projeto DIFERENTES em dois pontos-chave.
# A rede AQUI PREDIZ O ESTADO ABSOLUTO diretamente (nada de delta ou
# conexão residual) — igual ao script original nesse aspecto. O
# carregamento de dados, a normalização e a avaliação por rollout em
# grade também foram mantidos iguais de propósito, para que os dois
# scripts fiquem comparáveis lado a lado (mesmo dado, mesma métrica).
#
# DIFERENÇA 1 — ARQUITETURA: recorrência em vez de janela achatada
#   A versão MLP transforma a janela de K passos num vetor único
#   (K*18) e joga numa rede feed-forward. Aqui, a janela é tratada
#   como uma SEQUÊNCIA: uma GRU lê os K passos em ordem e produz um
#   resumo (hidden state) do histórico recente do sistema; esse resumo
#   (mais as massas) alimenta uma cabeça que prediz o próximo estado
#   absoluto diretamente. Isso deixa explícito para a rede que existe
#   uma ordem temporal entre os passos — a MLP não sabe disso, só vê
#   números concatenados.
#
# DIFERENÇA 2 — TREINO MULTI-PASSO: currículo de horizonte crescente
#   em vez de scheduled sampling com probabilidade fixa
#   A versão MLP, a cada passo do desenrolamento, sorteia se usa o
#   dado real ou a própria predição (probabilidade fixa de 50%). Aqui
#   a estratégia é outra: o desenrolamento é SEMPRE 100% autoregressivo
#   (a rede só vê suas próprias predições, nunca o dado real durante o
#   desenrolamento), mas o TAMANHO do horizonte cresce ao longo das
#   épocas — começa em 1 passo (quase teacher forcing puro) e vai até
#   HORIZONTE_MAX. A ideia: nas primeiras épocas o modelo ainda erra
#   muito, então pedir horizontes longos autoregressivos só ensina a
#   propagar erro; conforme o modelo melhora, aumentar o horizonte
#   força a rede a lidar com o próprio erro acumulado aos poucos.
#
# Ambas as estratégias (scheduled sampling e currículo de horizonte)
# atacam o mesmo problema — exposure bias / discrepância entre treino
# (teacher forcing) e uso real (autoregressivo) — só que por caminhos
# diferentes. Comparar as duas é o ponto deste exercício.
# =====================================================================


# ---------------------------------------------------------------
# 1. CARREGAR DADOS (idêntico ao script anterior — mesmo formato de
#    arquivo .npz, então não há motivo para mudar aqui)
# ---------------------------------------------------------------
def carregarTrajetorias(pastaNPZ):
    trajetorias, massasLista = [], []
    for arquivo in sorted(Path(pastaNPZ).glob("*.npz")):
        dados = np.load(arquivo)
        estado = dados["estado"]
        if estado.ndim < 2 or estado.shape[0] == 0:
            print(f"[aviso] pulando {arquivo.name}: trajetória vazia")
            continue
        estado = estado[:, :18]
        trajetorias.append(estado)
        massasLista.append(dados["massas"])
    return trajetorias, massasLista


# ---------------------------------------------------------------
# 2. NORMALIZAÇÃO (idêntica — z-score coluna a coluna)
# ---------------------------------------------------------------
def calcularNormalizacao(trajetorias):
    todos = np.concatenate(trajetorias)
    media = todos.mean(axis=0)
    desvio = todos.std(axis=0)
    desvio[desvio < 1e-8] = 1.0
    return media, desvio


def normalizar(x, media, desvio):
    return (x - media) / desvio


def desnormalizar(x, media, desvio):
    return x * desvio + media


# ---------------------------------------------------------------
# 3a. DATASET DE TREINO: janela de K passos -> HORIZONTE_MAX passos
#     futuros reais (o currículo escolhe, no loop de treino, quantos
#     desses passos usar em cada época)
# ---------------------------------------------------------------
class SequenciaComHorizonte(Dataset):
    def __init__(self, trajetorias, massas, tamanhoJanela, horizonteMax, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.horizonteMax = horizonteMax
        self.trajetorias = [normalizar(traj, media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]

        comprimentoTotal = tamanhoJanela + horizonteMax
        self.indices = []
        for idxTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - comprimentoTotal + 1
            for i in range(max(nAmostras, 0)):
                self.indices.append((idxTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idxTraj, i = self.indices[idx]
        traj = self.trajetorias[idxTraj]
        janela = traj[i:i + self.tamanhoJanela]
        futuros = traj[i + self.tamanhoJanela:i + self.tamanhoJanela + self.horizonteMax]
        massa = self.massas[idxTraj]
        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massa, dtype=t.float32),
                t.tensor(futuros, dtype=t.float32))


# ---------------------------------------------------------------
# 3b. DATASET DE VALIDAÇÃO: janela -> 1 passo (mesmo critério do
#     script anterior, para as duas versões serem comparáveis)
# ---------------------------------------------------------------
class SequenciaProximoEstado(Dataset):
    def __init__(self, trajetorias, massas, tamanhoJanela, media, desvio, mediaMassa, desvioMassa):
        self.tamanhoJanela = tamanhoJanela
        self.trajetorias = [normalizar(traj, media, desvio) for traj in trajetorias]
        self.massas = [normalizar(m, mediaMassa, desvioMassa) for m in massas]

        self.indices = []
        for idxTraj, traj in enumerate(self.trajetorias):
            nAmostras = len(traj) - tamanhoJanela
            for i in range(max(nAmostras, 0)):
                self.indices.append((idxTraj, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idxTraj, i = self.indices[idx]
        traj = self.trajetorias[idxTraj]
        janela = traj[i:i + self.tamanhoJanela]
        alvo = traj[i + self.tamanhoJanela]
        massa = self.massas[idxTraj]
        return (t.tensor(janela, dtype=t.float32),
                t.tensor(massa, dtype=t.float32),
                t.tensor(alvo, dtype=t.float32))


# ---------------------------------------------------------------
# 4. MODELO: GRU + cabeça que prediz o estado absoluto (em vez de
#    MLP feed-forward sobre a janela achatada)
# ---------------------------------------------------------------
class GRURecorrente(t.nn.Module):
    """
    A janela de K passos é lida em sequência por uma GRU (em vez de
    ser achatada num vetor único). O último hidden state da GRU é o
    "resumo" do histórico recente; concatenado com as massas, alimenta
    uma cabeça pequena (MLP de 1 camada oculta) que prediz o PRÓXIMO
    ESTADO ABSOLUTO diretamente — sem delta, sem conexão residual.
    """
    def __init__(self, dimEstado=18, dimMassa=3, hiddenGRU=64, hiddenCabeca=64, numCamadasGRU=1):
        super().__init__()
        self.gru = t.nn.GRU(input_size=dimEstado, hidden_size=hiddenGRU,
                             num_layers=numCamadasGRU, batch_first=True)
        self.cabeca = t.nn.Sequential(
            t.nn.Linear(hiddenGRU + dimMassa, hiddenCabeca),
            t.nn.ReLU(),
            t.nn.Linear(hiddenCabeca, dimEstado),
        )

    def forward(self, janela, massa):
        _, hFinal = self.gru(janela)        # hFinal: (numCamadasGRU, B, hiddenGRU)
        resumo = hFinal[-1]                 # (B, hiddenGRU) — última camada da GRU
        entradaCabeca = t.cat([resumo, massa], dim=1)
        return self.cabeca(entradaCabeca)   # (B, 18) — próximo estado absoluto, predito direto


# ---------------------------------------------------------------
# 5. TREINO COM CURRÍCULO DE HORIZONTE + AVALIAÇÃO
# ---------------------------------------------------------------
if __name__ == "__main__":
    TAMANHO_JANELA = 10
    HORIZONTE_MAX = 5      # horizonte final do currículo (mesmo valor do HORIZONTE_TREINO anterior)
    N_EPOCAS = 15
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    # Se TREINAR_MODELO = False, o treino é pulado e os pesos salvos em
    # CAMINHO_MODELO_SALVO são carregados direto — útil para testar
    # vários arquivos .npz na seção 6 sem precisar retreinar toda vez.
    TREINAR_MODELO = True
    CAMINHO_MODELO_SALVO = "gru_curriculo_horizonte.pt"

    trajetorias, massas = carregarTrajetorias("simulacoesArtificiais/simulacoes3C")

    nValidacao = max(1, len(trajetorias) // 5)
    trajetoriasTreino, trajetoriasValidacao = trajetorias[nValidacao:], trajetorias[:nValidacao]
    massasTreino, massasValidacao = massas[nValidacao:], massas[:nValidacao]

    media, desvio = calcularNormalizacao(trajetoriasTreino)
    mediaMassa = np.mean(massasTreino, axis=0)
    desvioMassa = np.std(massasTreino, axis=0)
    desvioMassa = np.where(desvioMassa < 1e-8, 1.0, desvioMassa)

    datasetTreino = SequenciaComHorizonte(trajetoriasTreino, massasTreino, TAMANHO_JANELA, HORIZONTE_MAX,
                                           media, desvio, mediaMassa, desvioMassa)
    datasetValidacao = SequenciaProximoEstado(trajetoriasValidacao, massasValidacao, TAMANHO_JANELA,
                                               media, desvio, mediaMassa, desvioMassa)

    loaderTreino = DataLoader(datasetTreino, batch_size=32, shuffle=True)
    loaderValidacao = DataLoader(datasetValidacao, batch_size=32, shuffle=False)

    # -----------------------------------------------------------
    # BASELINE DE PERSISTÊNCIA: "o próximo estado é igual ao último
    # da janela" (chute mais simples possível, sem rede nenhuma). Se a
    # loss da rede não for MELHOR do que isso, a rede não aprendeu
    # nada de útil — é o primeiro teste antes de discutir currículo,
    # arquitetura, caos, etc.
    # -----------------------------------------------------------
    criterioBaseline = t.nn.MSELoss()
    lossPersistencia = 0.0
    for janela, _massa, alvo in loaderValidacao:
        predPersistencia = janela[:, -1, :]  # "prevê" que nada muda
        lossPersistencia += criterioBaseline(predPersistencia, alvo).item()
    lossPersistencia /= len(loaderValidacao)
    print(f"Baseline de persistência (val, sem rede nenhuma): {lossPersistencia:.6f}")

    modelo = GRURecorrente().to(device)

    if TREINAR_MODELO:
        otimizador = t.optim.Adam(modelo.parameters(), lr=1e-3)
        criterio = t.nn.MSELoss()
        historicoLossVal = []

        for epoca in range(1, N_EPOCAS + 1):
            # Currículo: horizonte cresce linearmente de 1 até HORIZONTE_MAX
            # ao longo das épocas (na última época, horizonte = HORIZONTE_MAX).
            if N_EPOCAS > 1:
                horizonteAtual = 1 + (epoca - 1) * (HORIZONTE_MAX - 1) // (N_EPOCAS - 1)
            else:
                horizonteAtual = HORIZONTE_MAX

            modelo.train()
            lossTreino = 0.0
            for janela, massa, futuros in loaderTreino:
                janela = janela.to(device)      # (B, K, 18)
                massa = massa.to(device)        # (B, 3)
                futuros = futuros.to(device)    # (B, HORIZONTE_MAX, 18)

                otimizador.zero_grad()

                janelaAtual = janela
                lossPassos = []
                for h in range(horizonteAtual):
                    alvo = futuros[:, h, :]
                    pred = modelo(janelaAtual, massa)
                    lossPassos.append(criterio(pred, alvo))

                    # 100% autoregressivo dentro do desenrolamento: a
                    # janela sempre avança com a PRÓPRIA predição, nunca
                    # com o dado real (diferente do scheduled sampling).
                    janelaAtual = t.cat([janelaAtual[:, 1:, :], pred.detach().unsqueeze(1)], dim=1)

                loss = t.stack(lossPassos).mean()
                loss.backward()
                otimizador.step()

                lossTreino += loss.item()
            lossTreino /= len(loaderTreino)

            modelo.eval()
            lossVal = 0.0
            with t.no_grad():
                for janela, massa, alvo in loaderValidacao:
                    janela, massa, alvo = janela.to(device), massa.to(device), alvo.to(device)
                    predicao = modelo(janela, massa)
                    lossVal += criterio(predicao, alvo).item()
            lossVal /= len(loaderValidacao)
            historicoLossVal.append(lossVal)

            print(f"Época {epoca:3d} | horizonte: {horizonteAtual} | treino: {lossTreino:.6f} | val: {lossVal:.6f}")

        t.save(modelo.state_dict(), CAMINHO_MODELO_SALVO)

        # Sinal simples de subajuste: comparar a queda de loss nas
        # últimas épocas com a queda total do treino. Se ainda está
        # caindo forte no fim, mais épocas provavelmente ajudam ANTES
        # de investigar arquitetura ou culpar o caos do sistema.
        if len(historicoLossVal) >= 5:
            quedaRecente = historicoLossVal[-5] - historicoLossVal[-1]
            quedaTotal = historicoLossVal[0] - historicoLossVal[-1]
            if quedaTotal > 0 and quedaRecente / quedaTotal > 0.15:
                print(f"[aviso] a loss de validação ainda caiu {quedaRecente:.6f} nas últimas 5 épocas "
                      f"({100 * quedaRecente / quedaTotal:.0f}% da queda total) — o treino provavelmente "
                      f"ainda não convergiu. Considere aumentar N_EPOCAS antes de tirar conclusões sobre "
                      f"o horizonte de previsibilidade.")
        if lossVal >= lossPersistencia:
            print(f"[aviso] loss da rede ({lossVal:.6f}) não é melhor do que a baseline de persistência "
                  f"({lossPersistencia:.6f}) — algo no treino está errado antes de discutir caos.")
    else:
        modelo.load_state_dict(t.load(CAMINHO_MODELO_SALVO, map_location=device))
        print(f"Treino pulado — pesos carregados de {CAMINHO_MODELO_SALVO}")

    # -----------------------------------------------------------
    # Rollout autoregressivo — a interface é idêntica ao script
    # anterior (janela -> próximo estado), então a função de rollout
    # e o gráfico em grade são reaproveitados sem nenhuma mudança.
    # A diferença toda está dentro de `modelo(...)`.
    # -----------------------------------------------------------
    def rollout(modelo, trajetoria, massa, tamanhoJanela, nPassos, device):
        modelo.eval()
        trajNorm = normalizar(trajetoria, media, desvio)
        massaNorm = normalizar(massa, mediaMassa, desvioMassa)
        massaTensor = t.tensor(massaNorm, dtype=t.float32).unsqueeze(0).to(device)

        janelaAtual = trajNorm[:tamanhoJanela].copy()
        previsoes = []
        with t.no_grad():
            for _ in range(nPassos):
                janelaTensor = t.tensor(janelaAtual, dtype=t.float32).unsqueeze(0).to(device)
                pred = modelo(janelaTensor, massaTensor).squeeze(0).cpu().numpy()
                previsoes.append(pred)
                janelaAtual = np.vstack([janelaAtual[1:], pred])
        return desnormalizar(np.array(previsoes), media, desvio)

    nTrajetoriasTeste = len(trajetoriasValidacao)

    # Com muitas trajetórias de validação (dado real costuma ter dezenas),
    # colocar TODAS numa grade só vira ilegível — a janela interativa
    # espreme tudo e sobrepõe título/legenda de cada painel. A grade
    # mostra só uma amostra; as estatísticas de erro abaixo continuam
    # sendo calculadas sobre TODAS as trajetórias de validação.
    MAX_TRAJETORIAS_GRADE = 12
    indicesParaPlotar = list(range(min(nTrajetoriasTeste, MAX_TRAJETORIAS_GRADE)))
    nColunas = min(3, len(indicesParaPlotar)) if indicesParaPlotar else 1
    nLinhas = int(np.ceil(len(indicesParaPlotar) / nColunas)) if indicesParaPlotar else 1

    fig, eixos = plt.subplots(nLinhas, nColunas, figsize=(5 * nColunas, 4 * nLinhas), squeeze=False)

    errosMedios = []
    for idx in range(nTrajetoriasTeste):
        trajTeste = trajetoriasValidacao[idx]
        massaTeste = massasValidacao[idx]
        nPassos = min(300, len(trajTeste) - TAMANHO_JANELA)

        if nPassos <= 0:
            continue

        previsoes = rollout(modelo, trajTeste, massaTeste, TAMANHO_JANELA, nPassos, device)
        alvos = trajTeste[TAMANHO_JANELA:TAMANHO_JANELA + nPassos]

        erro = np.linalg.norm(previsoes - alvos, axis=1)
        errosMedios.append(erro.mean())

        if idx in indicesParaPlotar:
            linha, coluna = divmod(idx, nColunas)
            ax = eixos[linha][coluna]
            ax.plot(alvos[:, 0], label="Real (corpo 1, X)")
            ax.plot(previsoes[:, 0], "--", label="Predito (corpo 1, X)")
            ax.set_title(f"Trajetória {idx} | erro médio: {erro.mean():.3f}", fontsize=10)
            ax.set_xlabel("Passo")
            ax.set_ylabel("Posição X")
            ax.grid()
            if idx == indicesParaPlotar[0]:
                ax.legend(fontsize=8)  # legenda só no 1º painel, repetir em todos só polui

    for idx in range(len(indicesParaPlotar), nLinhas * nColunas):
        linha, coluna = divmod(idx, nColunas)
        eixos[linha][coluna].set_visible(False)

    tituloExtra = (f" (mostrando {len(indicesParaPlotar)} de {nTrajetoriasTeste} trajetórias)"
                   if nTrajetoriasTeste > len(indicesParaPlotar) else "")
    fig.suptitle(f"Rollout com GRU + currículo de horizonte (até {HORIZONTE_MAX}) — validação{tituloExtra}")
    plt.tight_layout()

    CAMINHO_FIGURA_GRADE = "rollout_validacao.png"
    plt.savefig(CAMINHO_FIGURA_GRADE, dpi=150)
    print(f"Figura da grade salva em: {CAMINHO_FIGURA_GRADE} (abra o arquivo para ver sem espremer a janela)")
    plt.show()

    print(f"\nErro médio entre trajetórias ({nTrajetoriasTeste} no total): {np.mean(errosMedios):.4f}")
    print(f"Desvio padrão entre trajetórias: {np.std(errosMedios):.4f}")
    print(f"Erro mínimo / máximo           : {np.min(errosMedios):.4f} / {np.max(errosMedios):.4f}")

    # -----------------------------------------------------------
    # 6. TESTE EM UM ARQUIVO .npz ESPECÍFICO (real x previsto)
    # -----------------------------------------------------------
    # Edite ARQUIVO_TESTE para apontar para qualquer simulação — pode
    # ser um arquivo de treino, de validação, ou uma simulação nova que
    # nunca foi vista pela rede — e rode o script de novo (com
    # TREINAR_MODELO = False depois da primeira vez, para não perder
    # tempo retreinando) para ver a trajetória real sobreposta à
    # trajetória prevista pela rede, corpo a corpo.
    ARQUIVO_TESTE = "simulacoesArtificiais/simulacoes3C/sim_0.npz"  # None para pular esta seção
    NUM_CORPOS = 3  # assume estado = [corpo1: x,y,z,vx,vy,vz, corpo2: ..., corpo3: ...]
    N_PASSOS_TESTE = 300  # sistema de 3 corpos é CAÓTICO: rollout muito longo (ex.: 10000
    # passos) só mostra ruído depois que o erro satura, sem informação útil. 300 passos dá
    # um trecho onde ainda dá para ver a divergência acontecendo. Use None para o arquivo
    # inteiro se quiser ver onde exatamente a previsão "explode".

    def testarArquivoNPZ(modelo, caminhoNPZ, tamanhoJanela, device, nPassos=None):
        """
        Carrega a trajetória de UM arquivo .npz, roda o rollout
        autoregressivo do modelo a partir dos primeiros `tamanhoJanela`
        passos reais, e devolve (real, previsto) já desnormalizados,
        alinhados passo a passo para comparação direta.
        """
        dados = np.load(caminhoNPZ)
        estado = dados["estado"][:, :18]
        massa = dados["massas"]

        if nPassos is None:
            nPassos = len(estado) - tamanhoJanela
        nPassos = min(nPassos, len(estado) - tamanhoJanela)
        if nPassos <= 0:
            raise ValueError(f"Trajetória de {caminhoNPZ} é curta demais para janela={tamanhoJanela}")

        previsto = rollout(modelo, estado, massa, tamanhoJanela, nPassos, device)
        real = estado[tamanhoJanela:tamanhoJanela + nPassos]
        return real, previsto

    def horizontePrevisibilidade(real, previsto):
        """
        Sistemas caóticos (como o de 3 corpos) amplificam qualquer erro
        exponencialmente — depois de um certo nº de passos, nem o
        melhor modelo do mundo acompanha a trajetória exata. Uma
        métrica mais honesta que "erro no passo 10000" é: até que passo
        o erro da previsão fica MENOR do que a variação natural da
        própria trajetória (uma linha de base tipo "chutar a média")?
        Depois desse ponto, a previsão de posição exata deixa de ter
        informação útil — o que é esperado, não um defeito da rede.
        """
        escalaReferencia = np.linalg.norm(real - real.mean(axis=0), axis=1).mean()
        erro = np.linalg.norm(real - previsto, axis=1)
        acimaDoLimiar = np.where(erro > escalaReferencia)[0]
        horizonte = acimaDoLimiar[0] if len(acimaDoLimiar) > 0 else len(erro)
        return horizonte, escalaReferencia

    def plotarComparacaoArquivo(real, previsto, nomeArquivo, numCorpos=3):
        """
        Uma linha por corpo: à esquerda, a trajetória no plano XY (real
        cheio, previsto tracejado, com marcador no ponto de partida); à
        direita, o erro de posição (norma) ao longo do tempo. Assume
        que cada corpo ocupa 6 colunas consecutivas: [x, y, z, vx, vy, vz].
        """
        fig, eixos = plt.subplots(numCorpos, 2, figsize=(10, 4 * numCorpos), squeeze=False)
        passos = np.arange(len(real))

        for corpo in range(numCorpos):
            col0 = corpo * 6
            xReal, yReal = real[:, col0], real[:, col0 + 1]
            xPrev, yPrev = previsto[:, col0], previsto[:, col0 + 1]
            erroPos = np.linalg.norm(real[:, col0:col0 + 3] - previsto[:, col0:col0 + 3], axis=1)

            axTraj = eixos[corpo][0]
            axTraj.plot(xReal, yReal, label="Real")
            axTraj.plot(xPrev, yPrev, "--", label="Previsto")
            axTraj.scatter([xReal[0]], [yReal[0]], color="green", zorder=5, label="Início")
            axTraj.set_title(f"Corpo {corpo + 1} — trajetória (plano XY)")
            axTraj.set_xlabel("X")
            axTraj.set_ylabel("Y")
            axTraj.axis("equal")
            axTraj.grid()
            axTraj.legend(fontsize=8)

            axErro = eixos[corpo][1]
            axErro.plot(passos, erroPos)
            axErro.set_title(f"Corpo {corpo + 1} — erro de posição (norma) ao longo do tempo")
            axErro.set_xlabel("Passo")
            axErro.set_ylabel("Erro")
            axErro.grid()

        fig.suptitle(f"Real x Previsto — {Path(nomeArquivo).name}")
        plt.tight_layout()
        plt.show()

    if ARQUIVO_TESTE is not None:
        real, previsto = testarArquivoNPZ(modelo, ARQUIVO_TESTE, TAMANHO_JANELA, device, nPassos=N_PASSOS_TESTE)
        plotarComparacaoArquivo(real, previsto, ARQUIVO_TESTE, numCorpos=NUM_CORPOS)

        erroGlobal = np.linalg.norm(real - previsto, axis=1)
        horizonte, escalaReferencia = horizontePrevisibilidade(real, previsto)
        print(f"\nArquivo de teste: {ARQUIVO_TESTE}")
        print(f"Erro médio no trecho testado ({len(real)} passos): {erroGlobal.mean():.4f}")
        print(f"Erro no último passo simulado: {erroGlobal[-1]:.4f}")
        print(f"Horizonte de previsibilidade: ~{horizonte} passos "
              f"(erro ultrapassa a variação natural da trajetória, {escalaReferencia:.4f}, "
              f"a partir daí)")