import numpy as np
import scipy as sc
import matplotlib
from matplotlib import pyplot as plt
import odil
import argparse
import os

# load custom tools
import loadWakeData as ldw
import fdTools as fdt
import makeProblem as mkp

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--usx', type=int, default=4, help="Up-sampling in x for numerical grid")
    parser.add_argument('--usy', type=int, default=4, help="Up-sampling in y for numerical grid")
    parser.add_argument('--ust', type=int, default=8, help="Up-sampling in t for numerical grid")
    parser.add_argument('--dsx', type=int, default=4, help="Down-sampling in x for numerical grid")
    parser.add_argument('--dsy', type=int, default=4, help="Down-sampling in y for numerical grid")
    parser.add_argument('--dst', type=int, default=1, help="Down-sampling in t for numerical grid")
    parser.add_argument('--Ns', type=float, default=0.1, help="Sigma for added noise")
    parser.add_argument('--W_dat', type=float, default=1.0, help="Weight for data loss")
    parser.add_argument('--W_phys', type=float, default=2.0, help="Weight for physics loss")
    parser.add_argument('--W_u', type=float, default=1e-4, help="Weight for velocity (divergence) smoothing")
    parser.add_argument('--W_f', type=float, default=1e-6, help="Weight for force smoothing")
    parser.add_argument('--pval',type=float, default=0.1, help="percentage of training data reserved for validation")
    parser.add_argument('--tmin', type=float, default=0, help="minimum t value")
    parser.add_argument('--tmax', type=float, default=1, help="maximum t value")
    parser.add_argument('--xmin', type=float, default = 0.5, help="minimum x value")
    parser.add_argument('--xmax', type=float, default = 3, help="maximum x value")
    parser.add_argument('--ymin', type=float, default = -1, help="minimum y value")
    parser.add_argument('--ymax', type=float, default = 1, help="maximum x value")  
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir='out_fields')
    parser.set_defaults(echo=1)
    parser.set_defaults(frames=20,
                        plot_every=1000,
                        report_every=10,
                        history_every=10)
    parser.set_defaults(optimizer='lbfgs', bfgs_m=5, bfgs_maxls=20)
    parser.set_defaults(multigrid=1)
    parser.set_defaults(nlvl=5)
    return parser.parse_args()

# parse arguments 
args = parse_args()


# load low-res/corrupted data:
Data   = ldw.loadAllData(args.xmin,args.xmax,args.ymin,args.ymax,args.tmin,args.tmax,args.dsx,args.dsy,args.dst,args.Ns)

DataHR = ldw.loadAllData(args.xmin,args.xmax,args.ymin,args.ymax,args.tmin,args.tmax,1,1,args.dst,0)

def plot_func(problem, state, epoch, frame, cbinfo=None):
    directory = f'./Plots_{frame}'
    os.makedirs(directory, exist_ok = True)
    
    xn,yn,tn = problem.domain.points(loc='nnn')
    un = problem.domain.field(state,'un')
    vn = problem.domain.field(state,'vn')
    xc,yc,tc = problem.domain.points(loc='ccn')
    fxn = problem.domain.field(state,'fxn')
    fyn = problem.domain.field(state,'fyn')
    
    nplot = 6
    dj = max(Data.XX.shape[2]//nplot,1)
    
    for j in range(0,Data.XX.shape[2],dj):
       fig, ax = plt.subplots(nrows=1,ncols=3)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.UU[:,:,j],vmin=-1.5,vmax=1.5)
       p2=ax[1].pcolor(DataHR.XX[:,:,j*args.dst],DataHR.YY[:,:,j*args.dst],DataHR.UU[:,:,j*args.dst],vmin=-1.5,vmax=1.5)
       p3=ax[2].pcolor(xn[:,:,problem.extra.pad1[2]+j*args.ust],yn[:,:,problem.extra.pad1[2]+j*args.ust],un[:,:,problem.extra.pad1[2]+j*args.ust],vmin=-1.5,vmax=1.5)
       for j2 in range(len(ax)):
         ax[j2].set_xlim([args.xmin, args.xmax])
         ax[j2].set_ylim([args.ymin, args.ymax])
         ax[j2].set_aspect('equal', adjustable='box')
       plt.savefig(directory+f'/u_{j}.png',dpi=250)
       plt.close('all')
       
       fig, ax = plt.subplots(nrows=1,ncols=3)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.VV[:,:,j],vmin=-1,vmax=1)
       p2=ax[1].pcolor(DataHR.XX[:,:,j*args.dst],DataHR.YY[:,:,j*args.dst],DataHR.VV[:,:,j*args.dst],vmin=-1,vmax=1)
       p3=ax[2].pcolor(xn[:,:,problem.extra.pad1[2]+j*args.ust],yn[:,:,problem.extra.pad1[2]+j*args.ust],vn[:,:,problem.extra.pad1[2]+j*args.ust],vmin=-1,vmax=1)
       for j2 in range(len(ax)):
         ax[j2].set_xlim([args.xmin, args.xmax])
         ax[j2].set_ylim([args.ymin, args.ymax])
         ax[j2].set_aspect('equal', adjustable='box')
       plt.savefig(directory+f'/v_{j}.png',dpi=250)
       plt.close('all')
       
       fig, ax = plt.subplots(nrows=1,ncols=3)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.FX[:,:,j],vmin=-3,vmax=3)
       p2=ax[1].pcolor(DataHR.XX[:,:,j*args.dst],DataHR.YY[:,:,j*args.dst],DataHR.FX[:,:,j*args.dst],vmin=-3,vmax=3)
       p3=ax[2].pcolor(xn[:,:,problem.extra.pad1[2]+j*args.ust],yn[:,:,problem.extra.pad1[2]+j*args.ust],fxn[:,:,problem.extra.pad1[2]+j*args.ust],vmin=-3,vmax=3)
       for j2 in range(len(ax)):
         ax[j2].set_xlim([args.xmin, args.xmax])
         ax[j2].set_ylim([args.ymin, args.ymax])
         ax[j2].set_aspect('equal', adjustable='box')
       plt.savefig(directory+f'/fx_{j}.png',dpi=250)
       plt.close('all')
       
       fig, ax = plt.subplots(nrows=1,ncols=3)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.FY[:,:,j],vmin=-3,vmax=3)
       p2=ax[1].pcolor(DataHR.XX[:,:,j*args.dst],DataHR.YY[:,:,j*args.dst],DataHR.FY[:,:,j*args.dst],vmin=-3,vmax=3)
       p3=ax[2].pcolor(xn[:,:,problem.extra.pad1[2]+j*args.ust],yn[:,:,problem.extra.pad1[2]+j*args.ust],fyn[:,:,problem.extra.pad1[2]+j*args.ust],vmin=-3,vmax=3)
       for j2 in range(len(ax)):
         ax[j2].set_xlim([args.xmin, args.xmax])
         ax[j2].set_ylim([args.ymin, args.ymax])
         ax[j2].set_aspect('equal', adjustable='box')
       plt.savefig(directory+f'/fy_{j}.png',dpi=250)
       plt.close('all')
       
       wz=fdt.computeVorticity(xn[:,:,problem.extra.pad1[2]+j*args.ust],yn[:,:,problem.extra.pad1[2]+j*args.ust],un[:,:,problem.extra.pad1[2]+j*args.ust],vn[:,:,problem.extra.pad1[2]+j*args.ust])
       fig, ax = plt.subplots(nrows=1,ncols=3)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.WZ[:,:,j],vmin=-10,vmax=10)
       p2=ax[1].pcolor(DataHR.XX[:,:,j*args.dst],DataHR.YY[:,:,j*args.dst],DataHR.WZ[:,:,j*args.dst],vmin=-10,vmax=10)
       p3=ax[2].pcolor(xn[:,:,j*args.ust],yn[:,:,j*args.ust],wz,vmin=-10,vmax=10)
       for j2 in range(len(ax)):
         ax[j2].set_xlim([args.xmin, args.xmax])
         ax[j2].set_ylim([args.ymin, args.ymax])
         ax[j2].set_aspect('equal', adjustable='box')
       plt.savefig(directory+f'/wz_{j}.png',dpi=250)
       plt.close('all')
       
       wz=fdt.computeDiv(xn[:,:,problem.extra.pad1[2]+j*args.ust],yn[:,:,problem.extra.pad1[2]+j*args.ust],un[:,:,problem.extra.pad1[2]+j*args.ust],vn[:,:,problem.extra.pad1[2]+j*args.ust])
       fig, ax = plt.subplots(nrows=1,ncols=3)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],-Data.W_Z[:,:,j],vmin=-10,vmax=10)
       p2=ax[1].pcolor(DataHR.XX[:,:,j*args.dst],DataHR.YY[:,:,j*args.dst],-DataHR.W_Z[:,:,j*args.dst],vmin=-10,vmax=10)
       p3=ax[2].pcolor(xn[:,:,j*args.ust],yn[:,:,j*args.ust],wz,vmin=-10,vmax=10)
       for j2 in range(len(ax)):
         ax[j2].set_xlim([args.xmin, args.xmax])
         ax[j2].set_ylim([args.ymin, args.ymax])
         ax[j2].set_aspect('equal', adjustable='box')
       plt.savefig(directory+f'/divU_{j}.png',dpi=250)
       plt.close('all')
       
    # save output fields
    wzn = np.zeros(un.shape)
    for j in range(wzn.shape[2]):
        wzn[:,:,j] = fdt.computeVorticity(xn[:,:,j],yn[:,:,j],un[:,:,j],vn[:,:,j])
    
    mdic = {'x_sol':xn, 'y_sol':yn, 't_sol':tn, 'u_sol':un, 'v_sol':vn, 'wz_sol':wzn, 'fx_sol':fxn,'fy_sol':fyn}
    
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
    fxn = problem.domain.field(state,'fxn')
    fyn = problem.domain.field(state,'fyn')
    
    nx1,ny1,nt1=Data.UU.shape
    
    xmin = args.xmin
    xmax = args.xmax
    ymin = args.ymin
    ymax = args.ymax
    tmin = args.tmin
    tmax = args.tmax
    
    ixx=np.where(np.logical_and(xn[:,0,0]>=xmin-1e-6,xn[:,0,0]<=xmax+1e-6))[0]
    iyy=np.where(np.logical_and(yn[0,:,0]>=ymin-1e-6,yn[0,:,0]<=ymax+1e-6))[0] 
    itt=np.where(np.logical_and(tn[0,0,:]>=tmin-1e-6,tn[0,0,:]<=tmax+1e-6))[0]
    #ixx=np.where(np.logical_and(xn[:,0,0]>=xmin,xn[:,0,0]<=xmax))[0]
    #iyy=np.where(np.logical_and(yn[0,:,0]>=ymin,yn[0,:,0]<=ymax))[0] 
    #itt=np.where(np.logical_and(tn[0,0,:]>=tmin,tn[0,0,:]<=tmax))[0]
    
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
    fxn = np.array(fxn)
    fxn = fxn[ixx,:,:][:,iyy,:][:,:,itt]
    fyn = np.array(fyn)
    fyn = fyn[ixx,:,:][:,iyy,:][:,:,itt]
    
    ix1 = np.where(abs(DataHR.XX[:,0,0]-xn[0,0,0])<1e-5)[0][0]
    iy1 = np.where(abs(DataHR.YY[0,:,0]-yn[0,0,0])<1e-5)[0][0]
    it1 = np.where(abs(DataHR.TT[0,0,:]-tn[0,0,0])<1e-5)[0][0]
    
    ddx=args.dsx//args.usx
    ddy=args.dsy//args.usy
    #ddt=1//args.ust
    
    xt = DataHR.XX[ix1::ddx,iy1::ddy,it1::]
    yt = DataHR.YY[ix1::ddx,iy1::ddy,it1::]
    tt = DataHR.TT[ix1::ddx,iy1::ddy,it1::]
    
    ut = DataHR.UU[ix1::ddx,iy1::ddy,it1::]     
    vt = DataHR.VV[ix1::ddx,iy1::ddy,it1::]     
    fxt = DataHR.FX[ix1::ddx,iy1::ddy,it1::]     
    fyt = DataHR.FY[ix1::ddx,iy1::ddy,it1::]     
    
    xn = xn[:,:,::args.ust]
    yn = yn[:,:,::args.ust]
    tn = tn[:,:,::args.ust]
    
    un = un[:,:,::args.ust]
    vn = vn[:,:,::args.ust]
    fxn = fxn[:,:,::args.ust]
    fyn = fyn[:,:,::args.ust]
    
    eUtrue= np.sqrt(np.mean((un - ut)**2)) 
    eVtrue= np.sqrt(np.mean((vn - vt)**2))
    eFxtrue= np.sqrt(np.mean((fxn - fxt)**2))
    eFytrue= np.sqrt(np.mean((fyn - fyt)**2))
    
    
    domain = problem.domain
    extra = problem.extra
      
    # loss function based on measured data:
    un = problem.domain.field(state,'un')
    vn = problem.domain.field(state,'vn')
    fxn = problem.domain.field(state,'fxn')
    fyn = problem.domain.field(state,'fyn')
    
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
    history.append('eFxtrue', eFxtrue)
    history.append('eFytrue', eFytrue)
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
plt.plot(D[:,0],D[:,-15])
plt.yscale('log')
plt.savefig('loss.png')
plt.close('all')

plt.plot(D[:,0],D[:,-12:-8])
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



xn,yn,tn = problem.domain.points(loc='nnn')
un = problem.domain.field(state,'un')
vn = problem.domain.field(state,'vn')
fxn = problem.domain.field(state,'fxn')
fyn = problem.domain.field(state,'fyn')


# save output fields
wzn = np.zeros(un.shape)
for j in range(wzn.shape[2]):
     wzn[:,:,j] = fdt.computeVorticity(xn[:,:,j],yn[:,:,j],un[:,:,j],vn[:,:,j])

mdic = {'x_sol':xn, 'y_sol':yn, 't_sol':tn, 'u_sol':un, 'v_sol':vn, 'wz_sol':wzn, 'fx_sol':fxn,'fy_sol':fyn, 'x_dat':Data.XX, 'y_dat':Data.YY, 't_dat':Data.TT, 'u_dat':Data.UU, 'v_dat':Data.VV, 'wz_dat':Data.WZ, 'fx_dat':Data.FX,'fy_dat':Data.FY,'x_hr':DataHR.XX, 'y_hr':DataHR.YY, 't_hr':DataHR.TT, 'u_hr':DataHR.UU, 'v_hr':DataHR.VV, 'wz_hr':DataHR.WZ, 'fx_hr':DataHR.FX, 'fy_hr':DataHR.FY}

sc.io.savemat('solutionData.mat',mdic)

