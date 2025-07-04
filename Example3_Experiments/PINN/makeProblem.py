import numpy as np
import scipy as sc
import matplotlib
from matplotlib import pyplot as plt
import odil
import argparse

# load my files
import defOperator as op
import fdTools as fdt

def make_problem(args,Data):
    # range of initial data
    xmin = Data.XX[0,0,0]
    xmax = Data.XX[-1,0,0]
    ymin = Data.YY[0,0,0]
    ymax = Data.YY[0,-1,0]
    tmin = Data.TT[0,0,0]
    tmax = Data.TT[0,0,-1]
    dsize = [xmin,xmax,ymin,ymax,tmin,tmax]
    dtype = np.float64
    
    domain = odil.Domain(cshape=(2, 2, 2),
                         dimnames=['x', 'y', 't'],
                         lower=(xmin, ymin, tmin),
                         upper=(xmax, ymax, tmax),
                         dtype=dtype,
                         multigrid=args.multigrid,
                         mg_interp=args.mg_interp,
                         mg_axes=[True, True, True],
                         mg_nlvl=args.nlvl)
    
    x_inner,y_inner,t_inner = domain.random_inner(args.Nci)
    
    NN_arch = []
    for j in range(args.nHL):
      NN_arch.append(args.nNHL)
    
    print(NN_arch)
    
    state = odil.State()
    state.fields['nnet']    = domain.make_neural_net([3] + NN_arch + [4])
    #state.fields['u_net']  = domain.make_neural_net([3] + NN_arch + [1])
    #state.fields['v_net']  = domain.make_neural_net([3] + NN_arch + [1])
    #state.fields['fx_net'] = domain.make_neural_net([3] + NN_arch + [1])
    #state.fields['fy_net'] = domain.make_neural_net([3] + NN_arch + [1])
    
    
    #    fields={
    #        'u_net': domain.make_neural_net([2] + args.NN_arch + [1]),
    #        'v_net': domain.make_neural_net([2] + args.NN_arch + [1]),
    #        'p_net': domain.make_neural_net([2] + args.NN_arch + [1]),
    #    })
    
    operator = op.operator
    
    extra = argparse.Namespace()
    
    Xn= Data.XX.ravel()
    Yn= Data.YY.ravel()
    Tn= Data.TT.ravel()
    Un= Data.UU.ravel()
    Vn= Data.VV.ravel()
    
    ind = np.random.permutation(len(Xn))
    Xn=Xn[ind]
    Yn=Yn[ind]
    Tn=Tn[ind]
    Un=Un[ind]
    Vn=Vn[ind]
    
    npt = np.shape(Xn)[0]
    nval= int(npt*args.pval)
    extra.Xn = Xn[nval:]    
    extra.Yn = Yn[nval:]    
    extra.Tn = Tn[nval:]    
    extra.Un = Un[nval:]    
    extra.Vn = Vn[nval:]    
    
    extra.Xv = Xn[:nval]    
    extra.Yv = Yn[:nval]    
    extra.Tv = Tn[:nval]    
    extra.Uv = Un[:nval]    
    extra.Vv = Vn[:nval] 
    
    extra.Re = Data.Re
    extra.Weights = [args.W_dat,args.W_phys]
    extra.x_inner = x_inner
    extra.y_inner = y_inner
    extra.t_inner = t_inner
    state = domain.init_state(state)
    problem = odil.Problem(op.operator, domain, extra)
    return problem,state
    
