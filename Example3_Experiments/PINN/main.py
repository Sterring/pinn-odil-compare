import numpy as np
import scipy as sc
import matplotlib
from matplotlib import pyplot as plt
import odil
import argparse
import os

import loadPIVData as ldw
import fdTools as fdt
import makeProblem as mkp


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nci', type=int, default=10000, help="Number of interior collocation points")
    parser.add_argument('--nNHL',type=int,nargs="*",default=80,help="Number of neurons in hidden layer")
    parser.add_argument('--nHL',type=int,nargs="*",default=6,help="Number of hidden layers")
    parser.add_argument('--W_dat', type=float, default=1.0, help="Weight for data loss")
    parser.add_argument('--W_phys', type=float, default=1.0, help="Weight for physics loss")
    parser.add_argument('--xmin', type=int, default=1, help="minimum time")
    parser.add_argument('--xmax', type=int, default=5, help="maximum time")
    parser.add_argument('--ymin', type=int, default=-2.5, help="minimum time")
    parser.add_argument('--ymax', type=int, default=2.5, help="maximum time")
    parser.add_argument('--pval',type=float, default=0.1, help="percentage of training data reserved for validation")
    parser.add_argument('--nadam',type=int, default=5000, help="number of epochs using ADAM before switching to L-BFGS")
    parser.add_argument('--adamLR',type=float, default=0.01, help="learning rate for ADAM")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir='out_fields')
    parser.set_defaults(echo=1)
    parser.set_defaults(frames=500,
                        plot_every=1000,
                        report_every=10,
                        history_every=10)
    #parser.set_defaults(optimizer='adam_tf', lr=1e-3)
    #parser.set_defaults(optimizer='lbfgs', bfgs_m=5, bfgs_maxls=20)
    parser.set_defaults(multigrid=0)
    parser.set_defaults(nlvl=5)
    return parser.parse_args()

# parse arguments 
args = parse_args()

# load low-res/corrupted data:
Data = ldw.loadTrainingData()

# make HR x y z grid points:
dx = Data.XX[-2,0,0]-Data.XX[-1,0,0]
dy = Data.YY[0,-2,0]-Data.YY[0,-1,0]
dt = Data.TT[0,0,1]-Data.TT[0,0,0]

usx = 4
usy = 4
ust = 8

tmin = Data.TT[0,0,0]
tmax = Data.TT[0,0,-1]


xx = np.arange(args.xmin,args.xmax,dx/usx)
yy = np.arange(args.ymin,args.ymax,dx/usy)
tt = Data.TT[0,0,:]

XX,YY,TT = np.meshgrid(xx,yy,tt,indexing='ij')


def plot_func(problem, state, epoch, frame, cbinfo=None):
    directory = f'./Plots_{frame}'
    os.makedirs(directory, exist_ok = True)
    tf = problem.domain.mod.tf
    
    inputs = [XX,YY,TT]
    unn,vnn,fxnn,fynn = problem.domain.neural_net(state,'nnet')(*inputs)
    
    if Data.XX.shape[2]>6:
       dtp = int(np.ceil(Data.XX.shape[2])/6)
    else:    
       dtp = 1
    
    for j in range(0,Data.XX.shape[2],dtp):
       fig, ax = plt.subplots(nrows=1,ncols=2)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.UU[:,:,j],vmin=-1.5,vmax=1.5)
       #p2=ax[1].pcolor(XX[:,:,j],YY[:,:,j],UU[:,:,j],vmin=-1.5,vmax=1.5)
       p3=ax[1].pcolor(XX[:,:,j],YY[:,:,j],unn[:,:,j],vmin=-1.5,vmax=1.5)
       for j2 in range(len(ax)):
         ax[j2].set_xlim([args.xmin, args.xmax])
         ax[j2].set_ylim([args.ymin, args.ymax])
         ax[j2].set_aspect('equal', adjustable='box')
       plt.savefig(directory+f'/u_{j}.png',dpi=250)
       plt.close('all')
       
       fig, ax = plt.subplots(nrows=1,ncols=2)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.VV[:,:,j],vmin=-1,vmax=1)
       #p2=ax[1].pcolor(XX[:,:,j],YY[:,:,j],VV[:,:,j],vmin=-1,vmax=1)
       p3=ax[1].pcolor(XX[:,:,j],YY[:,:,j],vnn[:,:,j],vmin=-1,vmax=1)
       for j2 in range(len(ax)):
         ax[j2].set_xlim([args.xmin, args.xmax])
         ax[j2].set_ylim([args.ymin, args.ymax])
         ax[j2].set_aspect('equal', adjustable='box')
       plt.savefig(directory+f'/v_{j}.png',dpi=250)
       plt.close('all')
       
       fig, ax = plt.subplots(nrows=1,ncols=2)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.FX[:,:,j],vmin=-3,vmax=3)
       #p2=ax[1].pcolor(XX[:,:,j],YY[:,:,j],FX[:,:,j],vmin=-3,vmax=3)
       p3=ax[1].pcolor(XX[:,:,j],YY[:,:,j],fxnn[:,:,j],vmin=-3,vmax=3)
       for j2 in range(len(ax)):
         ax[j2].set_xlim([args.xmin, args.xmax])
         ax[j2].set_ylim([args.ymin, args.ymax])
         ax[j2].set_aspect('equal', adjustable='box')
       plt.savefig(directory+f'/fx_{j}.png',dpi=250)
       plt.close('all')
       
       fig, ax = plt.subplots(nrows=1,ncols=2)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.FY[:,:,j],vmin=-3,vmax=3)
       #p2=ax[1].pcolor(XX[:,:,j],YY[:,:,j],FY[:,:,j],vmin=-3,vmax=3)
       p3=ax[1].pcolor(XX[:,:,j],YY[:,:,j],fynn[:,:,j],vmin=-3,vmax=3)
       for j2 in range(len(ax)):
         ax[j2].set_xlim([args.xmin, args.xmax])
         ax[j2].set_ylim([args.ymin, args.ymax])
         ax[j2].set_aspect('equal', adjustable='box')
       plt.savefig(directory+f'/fy_{j}.png',dpi=250)
       plt.close('all')
       
       WZ=fdt.computeVorticity(Data.XX[:,:,j],Data.YY[:,:,j],Data.UU[:,:,j],Data.VV[:,:,j])
       #WZHR=fdt.computeVorticity(XX[:,:,j],YY[:,:,j],UU[:,:,j],VV[:,:,j])
       wzn=fdt.computeVorticity(XX[:,:,j],YY[:,:,j],unn[:,:,j],vnn[:,:,j])
       
       fig, ax = plt.subplots(nrows=1,ncols=2)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],WZ,vmin=-10,vmax=10)
       #p2=ax[1].pcolor(XX[:,:,j],YY[:,:,j],WZHR,vmin=-10,vmax=10)
       p3=ax[1].pcolor(XX[:,:,j],YY[:,:,j],wzn,vmin=-10,vmax=10)
       for j2 in range(len(ax)):
         ax[j2].set_xlim([args.xmin, args.xmax])
         ax[j2].set_ylim([args.ymin, args.ymax])
         ax[j2].set_aspect('equal', adjustable='box')
       plt.savefig(directory+f'/wz_{j}.png',dpi=250)
       plt.close('all')
    
    inputs = [XX,YY,TT]
    un,vn,fxn,fyn = problem.domain.neural_net(state,'nnet')(*inputs)
    wzn = np.zeros(un.shape)
    for j in range(wzn.shape[2]):
       wzn[:,:,j] = fdt.computeVorticity(XX[:,:,j],YY[:,:,j],un[:,:,j],vn[:,:,j])
    
    # save output data
    mdic = {'x_sol':XX, 'y_sol':YY, 't_sol':TT, 'u_sol':un, 'v_sol':vn, 'wz_sol':wzn, 'fx_sol':fxn, 'fy_sol':fyn}
    
    if epoch>0:
      sc.io.savemat(directory+'/solutionData.mat',mdic)
      
      D = np.loadtxt('train.csv',delimiter=',',skiprows=1)
      plt.plot(D[:,0],D[:,-11])
      plt.yscale('log')
      plt.savefig('loss.png')
      plt.close('all')
      
            
      plt.plot(D[:,0],D[:,-8:-4])
      plt.yscale('log')
      plt.savefig('datVal_error.png')
      plt.close('all')
      
      plt.plot(D[:,0],D[:,-4:])
      plt.yscale('log')
      plt.savefig('datVal_RMS_error.png')
      plt.close('all')

odil.setup_outdir(args)

# setup problem
problem,state=mkp.make_problem(args,Data)


# add a history function for error between true field and n-n
def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
       
    extra=problem.extra
    
    inputs = [extra.Xn,extra.Yn,extra.Tn]
    unn,vnn,fxnn,fynn = problem.domain.neural_net(state,'nnet')(*inputs)
    
    UDatLoss = np.mean(np.abs(unn - extra.Un))
    VDatLoss = np.mean(np.abs(vnn - extra.Vn))
    
    UDatLoss_RMS = np.sqrt(np.mean((unn - extra.Un)**2))
    VDatLoss_RMS = np.sqrt(np.mean((vnn - extra.Vn)**2))
    
    inputs = [extra.Xv,extra.Yv,extra.Tv]
    unn,vnn,fxnn,fynn = problem.domain.neural_net(state,'nnet')(*inputs)
    
    UValLoss = np.mean(np.abs(unn - extra.Uv))
    VValLoss = np.mean(np.abs(vnn - extra.Vv))
    
    UValLoss_RMS = np.sqrt(np.mean((unn - extra.Uv)**2))
    VValLoss_RMS = np.sqrt(np.mean((vnn - extra.Vv)**2))
    
    # Add current parameters to `train.csv`.
    history.append('UDatLoss',UDatLoss)
    history.append('VDatLoss',VDatLoss)
    history.append('UValLoss',UValLoss)
    history.append('VValLoss',VValLoss)
    history.append('UDatLoss_RMS',UDatLoss_RMS)
    history.append('VDatLoss_RMS',VDatLoss_RMS)
    history.append('UValLoss_RMS',UValLoss_RMS)
    history.append('VValLoss_RMS',VValLoss_RMS)


lbfgs_plotevery = args.plot_every
lbfgs_frames    = args.frames

args.optimizer='adam_tf'
args.lr=args.adamLR
args.outdir='adam_init'
args.plot_every=args.nadam
args.frames=1
args.epochs=args.plot_every*args.frames


odil.setup_outdir(args)

callback = odil.make_callback(problem,
                              args,
                              plot_func=plot_func,
                              history_func=history_func)

odil.util.optimize(args, args.optimizer, problem, state, callback)

args.optimizer='lbfgs'
bfgs_m=5
bfgs_maxls=20
args.outdir='lbfgs_final'
args.plot_every=lbfgs_plotevery
args.frames=lbfgs_frames

args.epochs=args.plot_every*args.frames

odil.setup_outdir(args)

callback = odil.make_callback(problem,
                              args,
                              plot_func=plot_func,
                              history_func=history_func)

odil.util.optimize(args, args.optimizer, problem, state, callback)


D = np.loadtxt('train.csv',delimiter=',',skiprows=1)
plt.plot(D[:,0],D[:,-11])
plt.yscale('log')
plt.savefig('loss.png')
plt.close('all')



plt.plot(D[:,0],D[:,-8:-4])
plt.yscale('log')
plt.savefig('datVal_error.png')
plt.close('all')

plt.plot(D[:,0],D[:,-4:])
plt.yscale('log')
plt.savefig('datVal_RMS_error.png')
plt.close('all')



xx = np.arange(args.xmin,args.xmax,dx/usx)
yy = np.arange(args.ymin,args.ymax,dx/usy)

tmin = Data.TT[0,0,0]
tmax = Data.TT[0,0,-1]
dt = Data.TT[0,0,1]-Data.TT[0,0,0]

tt = np.arange(tmin,tmax,dt/ust)

XX,YY,TT = np.meshgrid(xx,yy,tt,indexing='ij')

xn = XX
yn = YY
tn = TT


inputs = [XX,YY,TT]
un,vn,fxn,fyn = problem.domain.neural_net(state,'nnet')(*inputs)
wzn = np.zeros(un.shape)
for j in range(wzn.shape[2]):
     wzn[:,:,j] = fdt.computeVorticity(xn[:,:,j],yn[:,:,j],un[:,:,j],vn[:,:,j])

# save output data
mdic = {'x_sol':xn, 'y_sol':yn, 't_sol':tn, 'u_sol':un, 'v_sol':vn, 'wz_sol':wzn, 'fx_sol':fxn,'fy_sol':fyn, 'x_dat':Data.XX, 'y_dat':Data.YY, 't_dat':Data.TT, 'u_dat':Data.UU, 'v_dat':Data.VV, 'wz_dat':Data.WZ, 'fx_dat':Data.FX,'fy_dat':Data.FY}

sc.io.savemat('solutionData.mat',mdic)

# save neural network data
NN_arch = []
for j in range(args.nHL):
  NN_arch.append(args.nNHL)

mdic2= {'nnarch':NN_arch,'weights':state.fields['nnet'].weights,'biases':state.fields['nnet'].biases}


sc.io.savemat('nnData.mat',mdic2)
