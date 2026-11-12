import torch as pt
import numpy as np
import matplotlib.pyplot as plt


#variáveis
densidade = 1000.0 #kg/m³
g = 9.80665 #m/s²
C = 10000 #constante - total de energia do sistema é arbitrário

#equação de Bernoulli retornando a PRESSÃO em pascais
def func1(v,h):
    return C - (densidade*(np.square(v)))/2 - (g*densidade*h)

#agora vou gerar os dados sintéticos da função
qtde_dados = 4000
v=np.random.uniform(0,25,qtde_dados) #pq não random.random(N)? pq vai retornar sempre entre 0 e 1, sem ter o teto, o que me obrigaria a multiplicar por 15, por exemplo
#^^^ velocidades de 0 a 25 m/s ^^^
h=np.random.uniform(0,10,qtde_dados) #altura de 0 a 10 metros
p = func1(v,h)

#normalizando os dados gerados para evitar overshooting dos dados
#primeiro: subtrai a média para que os dados fiquem em torno de 0 (-12.5, +12.5)
#segundo: para manter a magnitude dos dados correta e em torno de 1 eu divido pelo dsvio padrão dos dados
v_normalizado = (v-np.mean(v))/np.std(v)
h_normalizado = (h-np.mean(h))/np.std(h)
p_normalizado = (p-np.mean(p))/np.std(p)

#converter para TENSORES - integrado no autograd e roda na GPU
x = pt.tensor(np.vstack((v_normalizado,h_normalizado)).T, dtype=pt.float32) 
y = pt.tensor(p_normalizado.reshape(-1,1), dtype=pt.float32)

modeloRedeNeural = pt.nn.Sequential(
    pt.nn.Linear(2,32),
    pt.nn.Tanh(),
    pt.nn.Linear(32,32),
    pt.nn.Tanh(),
    pt.nn.Linear(32,1)
)

otimizador=pt.optim.Adam(modeloRedeNeural.parameters(), lr=1e-3)
loss=pt.nn.MSELoss()

#treinamento da rede
for epoca in range(8000):
    otimizador.zero_grad()
    y_predicao = modeloRedeNeural(x)
    lossAux=loss(y_predicao, y)
    lossAux.backward()
    otimizador.step()
    if epoca%100==0:
        print(f"Epoca {epoca}, loss = {lossAux.item():.6f}")

#avaliação dos resultados
y_predicao=modeloRedeNeural(x).detach().numpy()


#mostrando diferenças com plot
plt.figure(figsize=(6,5))
plt.scatter(y, y_predicao, s=5, alpha=0.6)
plt.xlabel("Pressão real (normalizada)")
plt.ylabel("Pressão prevista pela rede")
plt.title("Aproximação da equação de Bernoulli")
plt.grid(True)
plt.show()