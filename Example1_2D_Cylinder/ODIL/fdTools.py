import numpy as np 
import scipy.io
import odil

def interpolateBoundaries(Q):
    Q[0,:] = 2*Q[1,:]  - Q[2,:]
    Q[-1,:]= 2*Q[-2,:] - Q[-3,:]
    Q[:,0] = 2*Q[:,1]  - Q[:,2]
    Q[:,-1]= 2*Q[:,-2] - Q[:,-3]
    return Q

def computeVorticity(X,Y,U,V):
    dx=X[1,0]-X[0,0]
    dy=Y[0,1]-Y[0,0]
    def stencil_roll(q):
        return [
            np.roll(q, shift=np.negative(s), axis=(0, 1))
            for s in [(-1,-1),(1,-1),(-1,1),(1,1)]
        ]
    def grad_q(q):
        q_st = stencil_roll(q)
        q_sw, q_se, q_nw, q_ne = q_st
        dqdx = (q_ne+q_se-q_nw-q_sw)/(4*dx)
        dqdy = (q_ne+q_nw-q_se-q_sw)/(4*dy)
        return dqdx,dqdy
    dudx,dudy=grad_q(U)
    dvdx,dvdy=grad_q(V)
    Wz = dvdx-dudy
    Wz = np.array(Wz)
    Wz=interpolateBoundaries(Wz)
    return Wz

def upsampleField(Q,usx,usy,ust,nPad,nlvl,dsize):
    nx,ny,nt = Q.shape
    nxb=nx+nPad*2
    nyb=ny+nPad*2
    ntb=nt+nPad*2    
    nx1 = (nx-1)*usx+1
    ny1 = (ny-1)*usy+1
    nt1 = (nt-1)*ust+1
    nx1b = (nxb-1)*usx+1
    ny1b = (nyb-1)*usy+1
    nt1b = (ntb-1)*ust+1
    
    # need (nx2-1) to be a factor of 2**(nlvl-1) for multigrid to work!
    scale = 2**(nlvl-1)
    x_pad2=scale - ((nx1b-1) % scale)
    y_pad2=scale - ((ny1b-1) % scale)
    t_pad2=scale - ((nt1b-1) % scale)
        
    nx2 = nx1b+x_pad2
    ny2 = ny1b+y_pad2
    nt2 = nt1b+t_pad2
    
    xp1 = nPad*usx + x_pad2//2
    yp1 = nPad*usy + y_pad2//2
    tp1 = nPad*ust + t_pad2//2

    pad1=[xp1,yp1,tp1]
    
    Q2  = np.zeros((nx2,ny2,nt2))
    #Q1  = np.zeros((nx2,ny2,nt2))
    #M1  = np.zeros((nx2,ny2,nt2))
    # matrix for performing interpolations:
    xmat = np.arange(0,usx+1,1)/usx
    ymat = np.arange(0,usy+1,1)/usy
    tmat = np.arange(0,ust+1,1)/ust
    Xm,Ym,Tm = np.meshgrid(xmat,ymat,tmat,indexing='ij')
    for j1 in range(nx-1):
        for j2 in range(ny-1):
           for j3 in range(nt-1):
                  Qswp = Q[j1,j2,j3]
                  Qsep = Q[j1+1,j2,j3]
                  Qnwp = Q[j1,j2+1,j3]
                  Qnep = Q[j1+1,j2+1,j3]
                  Qswf = Q[j1,j2,j3+1]
                  Qsef = Q[j1+1,j2,j3+1]
                  Qnwf = Q[j1,j2+1,j3+1]
                  Qnef = Q[j1+1,j2+1,j3+1]
                  # interpolate in T:
                  Qsw = Qswp*(1-Tm)+Qswf*(Tm)
                  Qse = Qsep*(1-Tm)+Qsef*(Tm)
                  Qnw = Qnwp*(1-Tm)+Qnwf*(Tm)
                  Qne = Qnep*(1-Tm)+Qnef*(Tm)
                  # interpolate in X:
                  Qs  = Qsw*(1-Xm)+Qse*(Xm)
                  Qn  = Qnw*(1-Xm)+Qne*(Xm)
                  # interpolate in Y:
                  Qi  = Qs*(1-Ym) + Qn*Ym
                  for k1 in np.arange(0,usx+1,1):
                      for k2 in np.arange(0,usy+1,1):
                         for k3 in np.arange(0,ust+1,1):
                             Q2[xp1+j1*usx+k1,yp1+j2*usy+k2,tp1+j3*ust+k3] = Qi[k1,k2,k3]
    
    # compute updated domain size
    xmin = dsize[0]
    xmax = dsize[1]
    ymin = dsize[2]
    ymax = dsize[3]
    tmin = dsize[4]
    tmax = dsize[5]
    xmin2 = xmin - xp1/(nx1-1) * (xmax-xmin)
    xmax2 = xmax + (nx2-nx1-xp1)/(nx1-1) * (xmax-xmin)
    ymin2 = ymin - yp1/(ny1-1) * (ymax-ymin)
    ymax2 = ymax + (ny2-ny1-yp1)/(ny1-1) * (ymax-ymin)
    tmin2 = tmin - tp1/(nt1-1) * (tmax-tmin)
    tmax2 = tmax + (nt2-nt1-tp1)/(nt1-1) * (tmax-tmin)
    dsize2= [xmin2,xmax2,ymin2,ymax2,tmin2,tmax2]
    for j1 in range(0,xp1):
        for j2 in range(yp1,yp1+ny1):
            for j3 in range(tp1,tp1+nt1):
                Q2[j1,j2,j3] = Q2[xp1,j2,j3]
    for j1 in range(xp1+nx1,nx2):
        for j2 in range(yp1,yp1+ny1):
            for j3 in range(tp1,tp1+nt1):
                Q2[j1,j2,j3] = Q2[xp1+nx1-1,j2,j3]
    for j1 in range(nx2):
        for j2 in range(0,yp1):
            for j3 in range(tp1,tp1+nt1):
                Q2[j1,j2,j3] = Q2[j1,yp1,j3]
    for j1 in range(nx2):
        for j2 in range(yp1+ny1,ny2):
            for j3 in range(tp1,tp1+nt1):
                Q2[j1,j2,j3] = Q2[j1,yp1+ny1-1,j3]
    for j1 in range(nx2):
        for j2 in range(ny2):
            for j3 in range(0,tp1):
                Q2[j1,j2,j3] = Q2[j1,j2,tp1]
    for j1 in range(nx2):
        for j2 in range(ny2):
            for j3 in range(tp1+nt1,nt2):
                Q2[j1,j2,j3] = Q2[j1,j2,tp1+nt1-1]
    return Q2,pad1,dsize2

