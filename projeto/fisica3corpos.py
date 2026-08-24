import numpy as np
from config import G,epsilon

def yoshida4ordem3Corpos(estado,dt,m1,m2,m3):#utiliza algumas vezes o velocity-verlet
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
    r3=estado[12:15]
    v3=estado[15:18]

    # PRIMEIRA PARTE
    #DRIFT
    r1+=c1*dt*v1
    r2+=c1*dt*v2
    r3+=c1*dt*v3

    a1,a2,a3=atualizaAceleracoes_posicoes3Corpos(r1,r2,r3,m1,m2,m3)
    #KICK
    v1+=d1*dt*a1
    v2+=d1*dt*a2
    v3+=d1*dt*a3

    # SEGUNDA PARTE
    #DRIFT
    r1+=c2*dt*v1
    r2+=c2*dt*v2
    r3+=c2*dt*v3

    a1,a2,a3=atualizaAceleracoes_posicoes3Corpos(r1,r2,r3,m1,m2,m3)
    #KICK
    v1+=d2*dt*a1
    v2+=d2*dt*a2
    v3+=d2*dt*a3

    # TERCEIRA PARTE
    #DRIFT
    r1+=c3*dt*v1
    r2+=c3*dt*v2
    r3+=c3*dt*v3

    a1,a2,a3=atualizaAceleracoes_posicoes3Corpos(r1,r2,r3,m1,m2,m3)
    #KICK
    v1+=d3*dt*a1
    v2+=d3*dt*a2
    v3+=d3*dt*a3

    # QUARTA PARTE
    #DRIFT
    r1+=c4*dt*v1
    r2+=c4*dt*v2
    r3+=c4*dt*v3

    return np.concatenate([r1,v1,r2,v2,r3,v3])

def atualizaAceleracoes_posicoes3Corpos(r1,r2,r3,m1,m2,m3):
    #vetor posição
    pr12=r2-r1
    pr13=r3-r1
    pr23=r3-r2
    #radicando
    rr12=pr12[0]**2 + pr12[1]**2 + pr12[2]**2 + epsilon**2
    rr13=pr13[0]**2 + pr13[1]**2 + pr13[2]**2 + epsilon**2
    rr23=pr23[0]**2 + pr23[1]**2 + pr23[2]**2 + epsilon**2
    #fazendo o inverso para poupar algumas divisões
    inv_r12=1/(rr12*np.sqrt(rr12))
    inv_r13=1/(rr13*np.sqrt(rr13))
    inv_r23=1/(rr23*np.sqrt(rr23))

    a1=G*(m2*pr12*inv_r12 + m3*pr13*inv_r13)
    a2=G*(-m1*pr12*inv_r12 + m3*pr23*inv_r23)
    a3=G*(-m1*pr13*inv_r13 - m2*pr23*inv_r23)

    return a1,a2,a3

def calculaEnergiaDoSistema3Corpos(estado,m1,m2,m3):
    x1,y1,z1,vx1,vy1,vz1,\
    x2,y2,z2,vx2,vy2,vz2,\
    x3,y3,z3,vx3,vy3,vz3=estado[:18]

    r1=np.array([x1,y1,z1])
    r2=np.array([x2,y2,z2])
    r3=np.array([x3,y3,z3])

    r12=np.sqrt((np.linalg.norm(r2-r1))**2 + epsilon**2)
    r13=np.sqrt((np.linalg.norm(r3-r1))**2 + epsilon**2)
    r23=np.sqrt((np.linalg.norm(r3-r2))**2 + epsilon**2)

    cinetica=((vx1**2+vy1**2+vz1**2)*m1/2+(vx2**2+vy2**2+vz2**2)*m2/2+(vx3**2+vy3**2+vz3**2)*m3/2)
    
    potencial=-G*(m1*m2/r12+m1*m3/r13+m2*m3/r23)
    
    return cinetica+potencial

def calculaMomLin3Corpos(m1,m2,m3,v1,v2,v3):
    return np.linalg.norm(m1*v1 + m2*v2 + m3*v3)

def calculaMomAng3Corpos(m1,m2,m3,r1,r2,r3,v1,v2,v3):
    return m1*np.cross(r1,v1)+m2*np.cross(r2,v2)+m3*np.cross(r3,v3)