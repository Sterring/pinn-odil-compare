import numpy as np
import scipy as sc
import matplotlib
from matplotlib import pyplot as plt
import odil
import argparse
import os

# load custom tools
import loadWake as ldw
import fdTools as fdt
import makeProblem as mkp

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dsx', type=int, default=4, help="Down-sampling in x for low-res data")
    parser.add_argument('--dsy', type=int, default=4, help="Down-sampling in y for low-res data")
    parser.add_argument('--dst', type=int, default=4, help="Down-sampling in t for low-res data")
    parser.add_argument('--usx', type=int, default=4, help="Up-sampling in x for numerical grid")
    parser.add_argument('--usy', type=int, default=4, help="Up-sampling in y for numerical grid")
    parser.add_argument('--ust', type=int, default=4, help="Up-sampling in t for numerical grid")
    parser.add_argument('--pval',type=float, default=0.1, help="percentage of training data reserved for validation")
    parser.add_argument('--initD',type=int, default=1, help="Use sample data for initialisation")
    parser.add_argument('--W_dat', type=float, default=1.0, help="Weight for data loss")
    parser.add_argument('--W_phys', type=float, default=5.0, help="Weight for physics loss")
    parser.add_argument('--W_reg', type=float, default=0.0001, help="Weight for smoothing loss")
    parser.add_argument('--tmin', type=int, default=0, help="minimum time")
    parser.add_argument('--tmax', type=int, default=5, help="maximum time")
    parser.add_argument('--sgma', type=float,default=0.1, help="Gaussian noise S.D")
    
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir='out_fields')
    parser.set_defaults(echo=1)
    parser.set_defaults(frames=50,
                        plot_every=100,
                        report_every=10,
                        history_every=10)
    #parser.set_defaults(optimizer='adam_tf', lr=1e-3)
    parser.set_defaults(optimizer='lbfgs', bfgs_m=20, bfgs_maxls=20)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(nlvl=5)
    return parser.parse_args()

# parse arguments 
args = parse_args()


# load low-res/corrupted data:
Data=ldw.prepareData(args.dsx,args.dsy,args.dst,args.tmin,args.tmax,args.sgma)
# load reference solution data:
Data_HR=ldw.prepareData(1,1,args.dst,args.tmin,args.tmax,0)

def plot_func(problem, state, epoch, frame, cbinfo=None):
    directory = f'./Plots_{frame}'
    os.makedirs(directory, exist_ok = True)
    
    xn,yn,tn = problem.domain.points(loc='nnn')
    un = problem.domain.field(state,'un')
    vn = problem.domain.field(state,'vn')
    xc,yc,tc = problem.domain.points(loc='ccn')
    pn = problem.domain.field(state,'pn')
    
    if Data.XX.shape[2]>12:
       dtp = int(np.ceil(Data.XX.shape[2])/8)
    else:    
       dtp = 1
    
    for j in range(0,Data.XX.shape[2],dtp):
       fig, ax = plt.subplots(nrows=3,ncols=1)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.UU[:,:,j],vmin=0,vmax=2)
       p2=ax[1].pcolor(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],Data_HR.UU[:,:,j],vmin=0,vmax=2)
       p3=ax[2].pcolor(xn[:,:,problem.extra.pad1[2]+j*args.ust],yn[:,:,problem.extra.pad1[2]+j*args.ust],un[:,:,problem.extra.pad1[2]+j*args.ust],vmin=0,vmax=2)
       ax[0].set_xlim([1, 8])
       ax[0].set_ylim([-2, 2])
       ax[1].set_xlim([1, 8])
       ax[1].set_ylim([-2, 2])
       ax[2].set_xlim([1, 8])
       ax[2].set_ylim([-2, 2])
       plt.savefig(directory+f'/u_{j}.png')
       plt.close('all')
       
       fig, ax = plt.subplots(nrows=3,ncols=1)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.VV[:,:,j],vmin=-1,vmax=1)
       p2=ax[1].pcolor(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],Data_HR.VV[:,:,j],vmin=-1,vmax=1)
       p3=ax[2].pcolor(xn[:,:,problem.extra.pad1[2]+j*args.ust],yn[:,:,problem.extra.pad1[2]+j*args.ust],vn[:,:,problem.extra.pad1[2]+j*args.ust],vmin=-1,vmax=1)
       ax[0].set_xlim([1, 8])
       ax[0].set_ylim([-2, 2])
       ax[1].set_xlim([1, 8])
       ax[1].set_ylim([-2, 2])
       ax[2].set_xlim([1, 8])
       ax[2].set_ylim([-2, 2])
       plt.savefig(directory+f'/v_{j}.png')
       plt.close('all')
       
       fig, ax = plt.subplots(nrows=3,ncols=1)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.PP[:,:,j]-np.mean(Data.PP[:,:,j]),vmin=-0.2,vmax=0.2)
       p2=ax[1].pcolor(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],Data_HR.PP[:,:,j]-np.mean(Data_HR.PP[:,:,j]),vmin=-0.2,vmax=0.2)
       p3=ax[2].pcolor(xn[:,:,problem.extra.pad1[2]+j*args.ust],yn[:,:,problem.extra.pad1[2]+j*args.ust],pn[:,:,problem.extra.pad1[2]+j*args.ust]-np.mean(pn[problem.extra.pad1[0]:(-1-problem.extra.pad1[0]+1),problem.extra.pad1[1]:(-1-problem.extra.pad1[1]+1),problem.extra.pad1[2]+j*args.ust]),vmin=-0.2,vmax=0.2)
       ax[0].set_xlim([1, 8])
       ax[0].set_ylim([-2, 2])
       ax[1].set_xlim([1, 8])
       ax[1].set_ylim([-2, 2])
       ax[2].set_xlim([1, 8])
       ax[2].set_ylim([-2, 2])
       plt.savefig(directory+f'/p_{j}.png')
       plt.close('all')
       WZ=fdt.computeVorticity(Data.XX[:,:,j],Data.YY[:,:,j],Data.UU[:,:,j],Data.VV[:,:,j])
       WZHR=fdt.computeVorticity(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],Data_HR.UU[:,:,j],Data_HR.VV[:,:,j])
       wz=fdt.computeVorticity(xn[:,:,problem.extra.pad1[2]+j*args.ust],yn[:,:,problem.extra.pad1[2]+j*args.ust],un[:,:,problem.extra.pad1[2]+j*args.ust],vn[:,:,problem.extra.pad1[2]+j*args.ust])
       fig, ax = plt.subplots(nrows=3,ncols=1)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],WZ,vmin=-2,vmax=2)
       p2=ax[1].pcolor(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],WZHR,vmin=-2,vmax=2)
       p3=ax[2].pcolor(xn[:,:,j*args.ust],yn[:,:,j*args.ust],wz,vmin=-2,vmax=2)
       ax[0].set_xlim([1, 8])
       ax[0].set_ylim([-2, 2])
       ax[1].set_xlim([1, 8])
       ax[1].set_ylim([-2, 2])
       ax[2].set_xlim([1, 8])
       ax[2].set_ylim([-2, 2])
       plt.savefig(directory+f'/wz_{j}.png')
       plt.close('all')
       
    # save output fields
    un = problem.domain.field(state,'un')
    vn = problem.domain.field(state,'vn')
    pn = problem.domain.field(state,'pn')
    xn,yn,tn = problem.domain.points(loc='nnn')
    wzn = np.zeros(un.shape)
    for j in range(wzn.shape[2]):
     wzn[:,:,j] = fdt.computeVorticity(xn[:,:,j],yn[:,:,j],un[:,:,j],vn[:,:,j])

    mdic = {'x_sol':xn, 'y_sol':yn, 't_sol':tn, 'u_sol':un, 'v_sol':vn, 'wz_sol':wzn, 'p_sol':pn}

    sc.io.savemat(directory+'/solutionData.mat',mdic)
    return


odil.setup_outdir(args)

# setup problem
problem,state=mkp.make_problem(args,Data)

#add a history function for error between true field and n-n
def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
    xn,yn,tn = problem.domain.points(loc='nnn')
    un = problem.domain.field(state,'un')
    vn = problem.domain.field(state,'vn')
    pn = problem.domain.field(state,'pn')
    
    nx1,ny1,nt1=Data.UU.shape
    
    xmin = 1
    xmax = 8
    ymin = -2
    ymax = 2
    tmin = args.tmin
    tmax = args.tmax
    
    ixx=np.where(np.logical_and(xn[:,0,0]>=xmin-1e-6,xn[:,0,0]<=xmax+1e-6))[0]
    iyy=np.where(np.logical_and(yn[0,:,0]>=ymin-1e-6,yn[0,:,0]<=ymax+1e-6))[0] 
    itt=np.where(np.logical_and(tn[0,0,:]>=tmin-1e-6,tn[0,0,:]<=tmax-1e-6))[0]
    
    xn = np.array(xn)
    xn = xn[ixx,:,:][:,iyy,:][:,:,itt]
    yn = np.array(yn)
    yn = yn[ixx,:,:][:,iyy,:][:,:,itt]
    tn = np.array(tn)
    tn = tn[ixx,:,:][:,iyy,:][:,:,itt]
    
    un = np.array(un)
    un = un[ixx,:,:][:,iyy,:][:,:,itt]
    vn = np.array(vn)
    vn = vn[ixx,:,:][:,iyy,:][:,:,itt]
    pn = np.array(pn)
    pn = pn[ixx,:,:][:,iyy,:][:,:,itt]
    
    ix1 = np.where(abs(Data_HR.XX[:,0,0]-xn[0,0,0])<1e-5)[0][0]
    iy1 = np.where(abs(Data_HR.YY[0,:,0]-yn[0,0,0])<1e-5)[0][0]
    it1 = np.where(abs(Data_HR.TT[0,0,:]-tn[0,0,0])<1e-5)[0][0]
 
    if args.usx<=args.dsx: 
      ddx=args.dsx//args.usx
      ddy=args.dsy//args.usy
      #ddt=1//args.ust
      
      
      xt = Data_HR.XX[ix1::ddx,iy1::ddy,it1::]
      yt = Data_HR.YY[ix1::ddx,iy1::ddy,it1::]
      tt = Data_HR.TT[ix1::ddx,iy1::ddy,it1::]
      
      ut = Data_HR.UU[ix1::ddx,iy1::ddy,it1::]     
      vt = Data_HR.VV[ix1::ddx,iy1::ddy,it1::]     
      pt = Data_HR.PP[ix1::ddx,iy1::ddy,it1::]     
    else:
      ddx=args.usx//args.dsx
      ddy=args.usy//args.dsy
      
      xn = xn[::ddx,::ddy,:]
      yn = yn[::ddx,::ddy,:]
      tn = tn[::ddx,::ddy,:]
      un = un[::ddx,::ddy,:]
      vn = vn[::ddx,::ddy,:]
      pn = pn[::ddx,::ddy,:]
      
      xt = Data_HR.XX[:,:,it1::]
      yt = Data_HR.YY[:,:,it1::]
      tt = Data_HR.TT[:,:,it1::]
      
      ut = Data_HR.UU[:,:,it1::]     
      vt = Data_HR.VV[:,:,it1::]     
      pt = Data_HR.PP[:,:,it1::] 
      
    
    xn = xn[:,:,::args.ust]
    yn = yn[:,:,::args.ust]
    tn = tn[:,:,::args.ust]
    
    un = un[:,:,::args.ust]
    vn = vn[:,:,::args.ust]
    pn = pn[:,:,::args.ust]
    
    for j in range(pn.shape[2]):
      pn[:,:,j] = pn[:,:,j] +  - np.mean(pn[:,:,j]) + np.mean(pt[:,:,j])
    
    eUtrue= np.sqrt(np.mean((un - ut)**2)) 
    eVtrue= np.sqrt(np.mean((vn - vt)**2))
    ePtrue= np.sqrt(np.mean((pn - pt)**2))
    #eFxtrue= np.sqrt(np.mean((vnn - Data_HR.VV)**2)) 
    #eFytrue= np.sqrt(np.mean((vnn - Data_HR.VV)**2)) 
    
    
    domain = problem.domain
    extra = problem.extra
      
    # loss function based on measured data:
    un = problem.domain.field(state,'un')
    vn = problem.domain.field(state,'vn')
    pn = problem.domain.field(state,'pn')
    
    # downsample to data size:
    nx,ny,nt=domain.size(loc='nnn')
    nx1,ny1,nt1=extra.Un.shape
    
    u2 = un[(extra.pad1[0]):(extra.pad1[0]+(nx1-1)*extra.usf[0]+1):extra.usf[0],(extra.pad1[1]):(extra.pad1[1]+(ny1-1)*extra.usf[1]+1):extra.usf[1],(extra.pad1[2]):(extra.pad1[2]+(nt1-1)*extra.usf[2]+1):extra.usf[2]]
    v2 = vn[(extra.pad1[0]):(extra.pad1[0]+(nx1-1)*extra.usf[0]+1):extra.usf[0],(extra.pad1[1]):(extra.pad1[1]+(ny1-1)*extra.usf[1]+1):extra.usf[1],(extra.pad1[2]):(extra.pad1[2]+(nt1-1)*extra.usf[2]+1):extra.usf[2]]
    
    UDatLoss = np.mean(np.abs(extra.Mask*extra.MW*(u2 - extra.Un)))
    VDatLoss = np.mean(np.abs(extra.Mask*extra.MW*(v2 - extra.Vn)))
    M2 = 1-extra.Mask
    MW2= np.prod(M2.shape)/np.sum(M2)
    UValLoss = np.mean(np.abs(M2*MW2*(u2 - extra.Un)))
    VValLoss = np.mean(np.abs(M2*MW2*(v2 - extra.Vn)))
    
    UDatLoss_RMS = np.sqrt(np.mean(extra.Mask*extra.MW*((u2 - extra.Un)**2)))
    VDatLoss_RMS = np.sqrt(np.mean(extra.Mask*extra.MW*((v2 - extra.Vn)**2)))
    UValLoss_RMS = np.sqrt(np.mean(M2*MW2*((u2 - extra.Un)**2)))
    VValLoss_RMS = np.sqrt(np.mean(M2*MW2*((v2 - extra.Vn)**2)))
        
    # Add current parameters to `train.csv`.
    history.append('eUtrue', eUtrue)
    history.append('eVtrue', eVtrue)
    history.append('ePtrue', ePtrue)
    history.append('UDatLoss',UDatLoss)
    history.append('VDatLoss',VDatLoss)
    history.append('UValLoss',UValLoss)
    history.append('VValLoss',VValLoss)
    history.append('UDatLoss_RMS',UDatLoss_RMS)
    history.append('VDatLoss_RMS',VDatLoss_RMS)
    history.append('UValLoss_RMS',UValLoss_RMS)
    history.append('VValLoss_RMS',VValLoss_RMS)

callback = odil.make_callback(problem,
                              args,
                              plot_func=plot_func,
                              history_func=history_func)


odil.util.optimize(args, args.optimizer, problem, state, callback)


D = np.loadtxt('train.csv',delimiter=',',skiprows=1)
plt.plot(D[:,0],D[:,-14])
plt.yscale('log')
plt.savefig('loss.png')
plt.close('all')

plt.plot(D[:,0],D[:,-11:-8])
plt.yscale('log')
plt.savefig('soln_error.png')
plt.close('all')

plt.plot(D[:,0],D[:,-8:-4])
plt.yscale('log')
plt.savefig('datVal_error.png')
plt.close('all')

plt.plot(D[:,0],D[:,-4:])
plt.yscale('log')
plt.savefig('datVal_RMS_error.png')
plt.close('all')

# save output fields
un = problem.domain.field(state,'un')
vn = problem.domain.field(state,'vn')
pn = problem.domain.field(state,'pn')
xn,yn,tn = problem.domain.points(loc='nnn')
wzn = np.zeros(un.shape)
for j in range(wzn.shape[2]):
     wzn[:,:,j] = fdt.computeVorticity(xn[:,:,j],yn[:,:,j],un[:,:,j],vn[:,:,j])

mdic = {'x_sol':xn, 'y_sol':yn, 't_sol':tn, 'u_sol':un, 'v_sol':vn, 'wz_sol':wzn, 'p_sol':pn, 'x_dat':Data.XX, 'y_dat':Data.YY, 't_dat':Data.TT, 'u_dat':Data.UU, 'v_dat':Data.VV, 'wz_dat':Data.WZ, 'p_dat':Data.PP,'x_hr':Data_HR.XX, 'y_hr':Data_HR.YY, 't_hr':Data_HR.TT, 'u_hr':Data_HR.UU, 'v_hr':Data_HR.VV, 'p_hr':Data_HR.PP}

sc.io.savemat('solutionData.mat',mdic)

