import numpy as np
from config import G,epsilon

def yoshida4ordem2Corpos(estado,dt,m1,m2):#utiliza algumas vezes o velocity-verlet
    w1=1/(2-2**(1/3))
    w0=-2**(1/3)/(2-2**(1/3))
    c1=c4=w1/2
    c2=c3=(w0+w1)/2
    d1=d3=w1
    d2=w0
    estado=estado.copy()

    r1=estado[0:3]
    v1=estado[3:6]
    r2=estado[6:9]
    v2=estado[9:12]

    # PRIMEIRA PARTE
    #DRIFT
    r1+=c1*dt*v1
    r2+=c1*dt*v2

    a1,a2=atualizaAceleracoes_posicoes2Corpos(r1,r2,m1,m2)
    #KICK
    v1+=d1*dt*a1
    v2+=d1*dt*a2
    
    # SEGUNDA PARTE
    #DRIFT
    r1+=c2*dt*v1
    r2+=c2*dt*v2

    a1,a2=atualizaAceleracoes_posicoes2Corpos(r1,r2,m1,m2)
    #KICK
    v1+=d2*dt*a1
    v2+=d2*dt*a2

    # TERCEIRA PARTE
    #DRIFT
    r1+=c3*dt*v1
    r2+=c3*dt*v2

    a1,a2=atualizaAceleracoes_posicoes2Corpos(r1,r2,m1,m2)
    #KICK
    v1+=d3*dt*a1
    v2+=d3*dt*a2

    # QUARTA PARTE
    #DRIFT
    r1+=c4*dt*v1
    r2+=c4*dt*v2

    return np.concatenate([r1,v1,r2,v2])

def atualizaAceleracoes_posicoes2Corpos(r1,r2,m1,m2):
    #vetor posição
    pr12=r2-r1

    #radicando
    rr12=pr12[0]**2 + pr12[1]**2 + pr12[2]**2 + epsilon**2

    #fazendo o inverso para poupar algumas divisões
    inv_r12=1/(rr12*np.sqrt(rr12))
    
    a1=G*(m2*pr12*inv_r12)
    a2=G*(-m1*pr12*inv_r12)

    return a1,a2

def calculaEnergiaDoSistema2Corpos(estado,m1,m2): #e momentos linear e angular
    x1,y1,z1,vx1,vy1,vz1,\
    x2,y2,z2,vx2,vy2,vz2=estado[:12] #tira o tempo do estado para que não tena uma variável inútil sendo colocada aqui

    r1=np.array([x1,y1,z1])
    r2=np.array([x2,y2,z2])

    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)
    
    cinetica=((vx1**2+vy1**2+vz1**2)*m1/2+(vx2**2+vy2**2+vz2**2)*m2/2)
    
    potencial=-G*(m1*m2/r12)
    
    return cinetica+potencial

def calculaMomLin2Corpos(m1,m2,v1,v2):
    return np.linalg.norm(m1*v1 + m2*v2)

def calculaMomAng2Corpos(m1,m2,r1,r2,v1,v2):
    return m1*np.cross(r1,v1)+m2*np.cross(r2,v2)