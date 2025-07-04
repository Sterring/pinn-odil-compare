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
    nPad = 2  # padding to avoid having data points located on boundaries.

    # range of initial data
    xmin = Data.XX[0,0,0]
    xmax = Data.XX[-1,0,0]
    ymin = Data.YY[0,0,0]
    ymax = Data.YY[0,-1,0]
    tmin = Data.TT[0,0,0]
    tmax = Data.TT[0,0,-1]
    
    dsize = [xmin,xmax,ymin,ymax,tmin,tmax]
    
    # filter data as initial condition
    UU=sc.ndimage.gaussian_filter(Data.UU,1)
    VV=sc.ndimage.gaussian_filter(Data.VV,1)
    FX=sc.ndimage.gaussian_filter(Data.FX,1)
    FY=sc.ndimage.gaussian_filter(Data.FY,1)
    
    Uus,pad1,dsize2 = fdt.upsampleField(UU,args.usx,args.usy,args.ust,nPad,args.nlvl,dsize)
    Vus,pad1,dsize2 = fdt.upsampleField(VV,args.usx,args.usy,args.ust,nPad,args.nlvl,dsize)
    Fxus,pad1,dsize2 = fdt.upsampleField(FX,args.usx,args.usy,args.ust,nPad,args.nlvl,dsize)
    Fyus,pad1,dsize2 = fdt.upsampleField(FY,args.usx,args.usy,args.ust,nPad,args.nlvl,dsize)
    
    dtype = np.float64
    nx,ny,nt = Uus.shape
    xmin2,xmax2,ymin2,ymax2,tmin2,tmax2 = dsize2
    
    domain = odil.Domain(cshape=(nx-1, ny-1, nt-1),
                         dimnames=['x', 'y', 't'],
                         lower=(xmin2, ymin2, tmin2),
                         upper=(xmax2, ymax2, tmax2),
                         dtype=dtype,
                         multigrid=args.multigrid,
                         mg_interp=args.mg_interp,
                         mg_axes=[True, True, True],
                         mg_nlvl=args.nlvl)
    
    from odil import Field
    
    xn,yn,tn = domain.points(loc='nnn')
    
    state = odil.State(
        fields={
            'un': Field(Uus, loc='nnn'),
            'vn': Field(Vus, loc='nnn'),
            'fxn': Field(Fxus, loc='nnn'),
            'fyn': Field(Fyus, loc='nnn'),
        })
    
    if domain.multigrid:
        odil.printlog('multigrid levels:', domain.mg_cshapes)    
    extra = argparse.Namespace()
    #extra.measuredData = Data
    extra.Un= Data.UU
    extra.Vn= Data.VV
    extra.Mask = np.random.choice((0,1),extra.Un.shape,p=(args.pval,1-args.pval))
    extra.MW = np.prod(extra.Mask.shape)/np.sum(extra.Mask)
    extra.Re= 3900
    extra.usf= [args.usx,args.usy,args.ust]
    extra.Weights = [args.W_dat,args.W_phys,args.W_u,args.W_f]
    extra.pad1=pad1
    state = domain.init_state(state)
    problem = odil.Problem(op.operator, domain, extra)
    return problem, state
