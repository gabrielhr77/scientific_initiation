import torch as t
import numpy as np
'''
a = t.tensor([[1,2],[3,4]], dtype=t.float32)
b = t.rand(2,2)
print(a+b)
'''

a=t.tensor([1,2,3],dtype=t.float32)

b=t.zeros((3,3))
c=t.ones((3,3))
d=t.eye(3)
e=t.arange(0,10,2)
f=t.linspace(0,1,5)
G=t.arange(1,13).reshape(3,4)
H=(t.rand((1,12))*10).reshape(3,4)

#print(a,"\n",b,"\n",c,"\n",dd,"\n",e,"\n",f)
#print(t.norm(d))
#print(G)
#print(H)
#print(H.reshape(-1))
#print(H)
#print(H.sum())
#print(H.sum(dim=1))
#print(H.max(dim=1))
#print(H.mean())
#print(c+a)
print(t.cuda.is_available())
print(t.cuda.device_count())
device = t.device("cuda" if t.cuda.is_available() else "cpu")

x = t.ones((3, 3), device=device)
print(x)
