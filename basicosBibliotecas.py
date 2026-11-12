import numpy as np
import torch
import torch.nn as nn

v=np.array([1,40,2,-1])
M=np.array([[22,12,8],[28,24,2]])

#achar os menor e maior valores
minval=np.min(v)
maxval=np.max(v)

minvalM=np.min(M)
minvalM1=np.min(M,axis=1)
minvalM2=np.min(M,axis=0)
maxvalM=np.max(M)
maxvalM1=np.max(M,axis=0)
maxvalM2=np.max(M,axis=1)

#achar a posição no array do maior e menor valores
posminval=np.argmin(v)
posmaxval=np.argmax(v)

posminvalM=np.argmin(M)
posmaxvalM=np.argmax(M)

print('ARRAY - 1D\nMin: %g (pos no array %g)\nMax: %g (pos no array %g)' %(minval,posminval,maxval,posmaxval))
print('MATRIZ - 2D\nMin: %g (pos no array %g)\nMax: %g (pos no array %g)' %(minvalM,posminvalM,maxvalM,posmaxvalM))
print(minvalM1)
print('\nmin coluna: ', minvalM1, '\nmin linha: ',minvalM2,'\n')
print('\nmax linha: ',maxvalM1,'\nmax coluna: ',maxvalM2,'\n')