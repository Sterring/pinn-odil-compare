import numpy as np 
import scipy.io

# user packages
import fdTools as fdt

class GridData:
    XX = None
    YY = None
    TT = None
    UU = None
    VV = None
    PP = None
    WZ = None

def prepareData(dsx,dsy,dst,tmin,tmax,NoiseSigma):
    # make sure downsampling factors are integers:
    def fixds(ds):
        ds = int(ds)
        if ds<=0:
            ds=1
        return ds
    dsx=fixds(dsx)
    dsy=fixds(dsy)
    dst=fixds(dst)
    # Get data file from: 
    #         https://github.com/maziarraissi/PINNs/tree/master/main/Data/cylinder_nektar_wake.mat
    data = scipy.io.loadmat('cylinder_nektar_wake.mat')
    #
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    #
    N = X_star.shape[0]
    T = t_star.shape[0]
    #
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    #
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    #
    # Reshape to 3D grid
    n1 = 100
    n2 = 50
    n3 = 200
    XX = np.reshape(XX,(n2,n1,n3))
    YY = np.reshape(YY,(n2,n1,n3))
    TT = np.reshape(TT,(n2,n1,n3))
    UU = np.reshape(UU,(n2,n1,n3))
    VV = np.reshape(VV,(n2,n1,n3))
    PP = np.reshape(PP,(n2,n1,n3))
    
    itt=np.where(np.logical_and(TT[0,0,:]>=tmin,TT[0,0,:]<=tmax))
    itt=itt[0]
        
    ix = np.arange(0,XX.shape[1],dsx)
    iy = np.arange(0,XX.shape[0],dsy)
    it = np.arange(itt[0],itt[-1],dst)
    
    XX = XX[iy,:,:][:,ix,:][:,:,it]
    YY = YY[iy,:,:][:,ix,:][:,:,it]
    TT = TT[iy,:,:][:,ix,:][:,:,it]
    UU = UU[iy,:,:][:,ix,:][:,:,it]
    VV = VV[iy,:,:][:,ix,:][:,:,it]
    PP = PP[iy,:,:][:,ix,:][:,:,it]
    
    # transpose to better shape
    XX=XX.transpose(1, 0, 2)
    YY=YY.transpose(1, 0, 2)
    TT=TT.transpose(1, 0, 2)
    UU=UU.transpose(1, 0, 2)
    VV=VV.transpose(1, 0, 2)
    PP=PP.transpose(1,0,2)
    
    # add Gaussian noise:
    UU = UU+np.random.normal(0, NoiseSigma, XX.shape)
    VV = VV+np.random.normal(0, NoiseSigma, XX.shape)
    WZ = np.zeros(XX.shape)
    for j in range(XX.shape[2]):
        WZ[:,:,j] = fdt.computeVorticity(XX[:,:,j],YY[:,:,j],UU[:,:,j],VV[:,:,j])
    
    Data = GridData()
    Data.XX = XX
    Data.YY = YY
    Data.TT = TT
    Data.UU = UU
    Data.VV = VV
    Data.PP = PP
    Data.WZ = WZ
    
    return Data

