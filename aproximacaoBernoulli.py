import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros físicos
rho = 1000.0  # densidade da água (kg/m³)
g = 9.81      # gravidade (m/s²)
C = 2e5       # constante de energia total (Pa)

# Função de Bernoulli: calcula pressão teórica
def bernoulli_pressure(v, h):
    return C - 0.5 * rho * v**2 - rho * g * h

# Gerando dados sintéticos
N = 2000
v = np.random.uniform(0, 15, N)     # velocidades de 0 a 15 m/s
h = np.random.uniform(0, 10, N)     # alturas de 0 a 10 m
P = bernoulli_pressure(v, h)        # pressão calculada

# Normalizando (para ajudar a rede)
v_norm = (v - np.mean(v)) / np.std(v)
h_norm = (h - np.mean(h)) / np.std(h)
P_norm = (P - np.mean(P)) / np.std(P)

# Convertendo para tensores
X = torch.tensor(np.vstack((v_norm, h_norm)).T, dtype=torch.float32)
y = torch.tensor(P_norm.reshape(-1, 1), dtype=torch.float32)

# Rede neural simples
model = nn.Sequential(
    nn.Linear(2, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Treinamento
for epoch in range(2000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

# Avaliação
y_pred = model(X).detach().numpy()

# Comparando valores reais vs previstos
plt.figure(figsize=(6,5))
plt.scatter(y, y_pred, s=5, alpha=0.6)
plt.xlabel("Pressão real (normalizada)")
plt.ylabel("Pressão prevista pela rede")
plt.title("Aproximação da equação de Bernoulli")
plt.grid(True)
plt.show()

