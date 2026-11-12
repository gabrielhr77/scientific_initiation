import numpy as np, matplotlib.pyplot as plt, scipy.stats as stats



#parametros
n1=30 #qntdd de elementos da popução 1 
n2=40  #qntdd de elementos da popução 2
mn1=1  #media pop 1
mn2=2  #media pop 2

#gerando os dados
data1=mn1+np.random.randn(n1)
data2=mn2+np.random.randn(n2)

t,p=stats.ttest_ind(data1,data2)
print(t)
print(p)

plt.plot(0+np.random.randn(n1)/15,data1,'ro')
plt.plot(1+np.random.randn(n2)/15,data2,'bs')
plt.xlim([-1,2])
plt.xticks([0,1],labels=['Group 1','Group 2'])
plt.show()

