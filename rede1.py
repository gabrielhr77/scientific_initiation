import numpy as np

def sigmoid(soma):
	return 1/(1+np.exp(-soma))

def sigmoidDerivada(sig):
	return sig*(1-sig)


#definição do XOR

entradas = np.array([[0,0],[0,1],[1,0],[1,1]])

saidas = np.array([[0],[1],[1],[0]])
'''
pesos0=np.array([[-0.424,-0.740,-0.961],[0.358,-0.577,-0.469]])

pesos1=np.array([[-0.017],[-0.893],[0.148]])
'''
np.random.seed(42)  # Para resultados reprodutíveis
pesos0 = 2 * np.random.rand(2, 3) - 1
pesos1 = 2 * np.random.rand(3, 1) - 1

tentativas=100000
taxaAprendizagem=0.3
escala=1


for j in range(tentativas):

	camadaEntrada=entradas
	somaSinapse0=np.dot(camadaEntrada,pesos0)
	camadaOculta=sigmoid(somaSinapse0)
	
	somaSinapse1=np.dot(camadaOculta,pesos1)
	camadaSaida=sigmoid(somaSinapse1)

	erroCamadaSaida=saidas-camadaSaida
	mediaAbsoluta=np.mean(np.abs(erroCamadaSaida))

	derivadaSaida=sigmoidDerivada(camadaSaida)
	deltaSaida=erroCamadaSaida*derivadaSaida

	pesos1Transposta=pesos1.T
	deltaSaidaXPeso=deltaSaida.dot(pesos1Transposta)
	deltaCamadaOculta=deltaSaidaXPeso*sigmoidDerivada(camadaOculta)
	
	camadaOcultaTransposta=camadaOculta.T
	pesosNovo1=camadaOcultaTransposta.dot(deltaSaida)
	pesos1=(pesos1*escala)+(pesosNovo1*taxaAprendizagem)

	entradaTransposta = camadaEntrada.T
	pesosNovo0 = entradaTransposta.dot(deltaCamadaOculta)
	pesos0 = (pesos0 * escala) + (pesosNovo0 * taxaAprendizagem)

	if j % 1000 == 0:
    		print(f'Tentativa {j} — Erro medio: {np.mean(np.abs(erroCamadaSaida))}')

print(pesos1)

print("\nSaída final da rede para cada entrada:")
for entrada, saida in zip(entradas, camadaSaida):
	print(f"{entrada} => {saida}")













