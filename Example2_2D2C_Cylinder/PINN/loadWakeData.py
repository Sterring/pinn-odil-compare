import numpy as np
import os
import scipy.io

import fdTools as fdt

Re = 3900

class GridData:
    XX = None
    YY = None
    TT = None
    UU = None
    VV = None
    WW = None
    PP = None
    WZ = None
    U_Z= None
    V_Z= None
    W_Z= None
    U_ZZ=None
    V_ZZ=None
    W_ZZ=None
    FX = None
    FY = None

def readDataFrame(j,xmin,xmax,ymin,ymax,dsx,dsy):
    DataDir = '../Data'
    Fname = DataDir+f'/vp{j:03d}.dat'
    data = np.loadtxt(Fname,delimiter=',')
    
    Datalist = []
    for i in range(data.shape[1]):
       Datalist.append(data[:,i])
    
    nx = int(np.max(Datalist[0]))
    ny = int(np.max(Datalist[1]))
        
    for i in range(len(Datalist)):
        Datalist[i] = Datalist[i].reshape((nx,ny))
    
    
    ixmin=np.where(Datalist[2][:,0]>=xmin)[0][0]
    ixmax=np.where(Datalist[2][:,0]<=xmax)[0][-1]+1
    iymin=np.where(Datalist[3][0,:]>=ymin)[0][0]
    iymax=np.where(Datalist[3][0,:]<=ymax)[0][-1]+1
    
    for i in range(len(Datalist)):
        Datalist[i] = Datalist[i][ixmin:ixmax:dsx,iymin:iymax:dsy]
    
    return Datalist


def loadAllData(xmin,xmax,ymin,ymax,tmin,tmax,dsx,dsy,dst,NoiseSigma):
    dt = 0.1
    dx = 0.01
    dy = 0.01
    itmin = int(tmin/dt)+1
    itmax = int(tmax/dt)+1
    itt  = np.arange(itmin,itmax+1e-4,dst)
    ixmin = int(xmin/dx)
    ixmax = int(xmax/dx)
    ixx  = np.arange(ixmin,ixmax+1e-4,dsx)
    iymin = int(ymin/dy)
    iymax = int(ymax/dy)
    iyy  = np.arange(iymin,iymax+1e-4,dsy)
    print(itt)
    print(iyy*dy)
    print(ixx*dx)
    nt = len(itt)
    nx = len(ixx)
    ny = len(iyy)
    
    T    = np.zeros((nx,ny,nt))
    X    = np.zeros((nx,ny,nt))
    Y    = np.zeros((nx,ny,nt))
    U    = np.zeros((nx,ny,nt))
    V    = np.zeros((nx,ny,nt))
    W    = np.zeros((nx,ny,nt))
    P    = np.zeros((nx,ny,nt))
    Wz   = np.zeros((nx,ny,nt))
    U_z  = np.zeros((nx,ny,nt))
    V_z  = np.zeros((nx,ny,nt))
    W_z  = np.zeros((nx,ny,nt))
    U_zz = np.zeros((nx,ny,nt))
    V_zz = np.zeros((nx,ny,nt))
    W_zz = np.zeros((nx,ny,nt))
    P_X = np.zeros((nx,ny,nt))
    P_Y = np.zeros((nx,ny,nt))
    
    print(X.shape)
    
    for j in range(len(itt)):
        DataList = readDataFrame(int(itt[j]),xmin,xmax,ymin,ymax,dsx,dsy)
        T[:,:,j] = itt[j]*dt
        X[:,:,j] = DataList[2]
        Y[:,:,j] = DataList[3]
        U[:,:,j] = DataList[4]
        V[:,:,j] = DataList[5]
        if (NoiseSigma==0) and (max(dsx,dsy)==1):
          W[:,:,j] = DataList[6]
          P[:,:,j] = DataList[7]
          Wz[:,:,j]= DataList[10]
          U_z[:,:,j] = DataList[11]
          V_z[:,:,j] = DataList[12]
          W_z[:,:,j] = DataList[13]
          U_zz[:,:,j] = DataList[14]
          V_zz[:,:,j] = DataList[15]
          W_zz[:,:,j] = DataList[16]
          dpdx,dpdy = fdt.computePressureGradient(DataList[2],DataList[3],DataList[7])
          P_X[:,:,j] = dpdx
          P_Y[:,:,j] = dpdy
    
    # add Gaussian noise:
    if NoiseSigma>0 or (max(dsx,dsy)>1):
       U = U+np.random.normal(0, NoiseSigma, X.shape)
       V = V+np.random.normal(0, NoiseSigma, X.shape)
       for j in range(len(itt)):
          Wz[:,:,j] = fdt.computeVorticity(X[:,:,j],Y[:,:,j],U[:,:,j],V[:,:,j])
    
    
    Data = GridData()
    Data.XX = X
    Data.YY = Y
    Data.TT = T
    Data.UU = U
    Data.VV = V
    Data.WW = W
    Data.PP = P
    Data.WZ = Wz
    Data.U_Z= U_z
    Data.V_Z= V_z
    Data.W_Z= W_z
    Data.U_ZZ=U_zz
    Data.V_ZZ=V_zz
    Data.W_ZZ=W_zz
    Data.FX  = -W*U_z-P_X+(1/Re)*U_zz
    Data.FY  = -W*V_z-P_Y+(1/Re)*U_zz
    return Data

#xmin = 1
#xmax = 4
#ymin = -1
#ymax = 1
#tmin = 0
#tmax = 10
#dsx  = 4
#dsy  = 4
#dst  = 1

#Data=loadAllData(xmin,xmax,ymin,ymax,tmin,tmax,dsx,dsy,dst)

#import matplotlib
#from matplotlib import pyplot as plt

#for i in range(X.shape[2]):
#    plt.pcolor(X[:,:,i],Y[:,:,i],Wz[:,:,i],vmin=-10,vmax=10)
#    plt.gca().set_aspect('equal')
#    plt.show()
#    plt.close('all') 

