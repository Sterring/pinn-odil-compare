import numpy as np
import scipy as sc
import matplotlib
from matplotlib import pyplot as plt
import odil
import argparse
import os

import loadWake as ldw
import fdTools as fdt
import makeProblem as mkp


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Nci', type=int, default=10000, help="Number of interior collocation points")
    parser.add_argument('--nNHL',type=int, default=20,help="Number of neurons in hidden layer")
    parser.add_argument('--nHL',type=int,  default=6, help="Number of hidden layers")
    parser.add_argument('--dsx', type=int, default=4, help="Down-sampling in x for low-res data")
    parser.add_argument('--dsy', type=int, default=4, help="Down-sampling in y for low-res data")
    parser.add_argument('--dst', type=int, default=4, help="Down-sampling in t for low-res data")
    parser.add_argument('--W_dat', type=float, default=1.0, help="Weight for data loss")
    parser.add_argument('--W_phys', type=float, default=1.0, help="Weight for physics loss")
    parser.add_argument('--tmin', type=int, default=0, help="minimum time")
    parser.add_argument('--tmax', type=int, default=5, help="maximum time")
    parser.add_argument('--sgma', type=float,default=0.1, help="Gaussian noise S.D")
    parser.add_argument('--pval',type=float, default=0.1, help="percentage of training data reserved for validation")
    parser.add_argument('--nadam',type=int, default=500, help="number of epochs using ADAM before switching to L-BFGS")
    parser.add_argument('--adamLR',type=float, default=0.01, help="learning rate for ADAM")
    odil.util.add_arguments(parser)
    odil.linsolver.add_arguments(parser)
    parser.set_defaults(outdir='out_fields')
    parser.set_defaults(echo=1)
    parser.set_defaults(frames=10,
                        plot_every=1000,
                        report_every=10,
                        history_every=1)
    parser.set_defaults(multigrid=0)
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
    tf = problem.domain.mod.tf
    
    inputs = [Data_HR.XX,Data_HR.YY,Data_HR.TT]
    unn,vnn,pnn = problem.domain.neural_net(state,'nnet')(*inputs)
        
    if Data.XX.shape[2]>12:
       dtp = int(np.ceil(Data.XX.shape[2])/8)
    else:    
       dtp = 1
    
    for j in range(0,Data.XX.shape[2],dtp):
       fig, ax = plt.subplots(nrows=3,ncols=1)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.UU[:,:,j],vmin=0,vmax=2)
       p2=ax[1].pcolor(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],Data_HR.UU[:,:,j],vmin=0,vmax=2)
       p3=ax[2].pcolor(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],unn[:,:,j],vmin=0,vmax=2)
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
       p3=ax[2].pcolor(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],vnn[:,:,j],vmin=-1,vmax=1)
       ax[0].set_xlim([1, 8])
       ax[0].set_ylim([-2, 2])
       ax[1].set_xlim([1, 8])
       ax[1].set_ylim([-2, 2])
       ax[2].set_xlim([1, 8])
       ax[2].set_ylim([-2, 2])
       plt.savefig(directory+f'/v_{j}.png')
       plt.close('all')
       
       fig, ax = plt.subplots(nrows=3,ncols=1)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],Data.PP[:,:,j],vmin=-0.2,vmax=0.2)
       p2=ax[1].pcolor(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],Data_HR.PP[:,:,j],vmin=-0.2,vmax=0.2)
       p3=ax[2].pcolor(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],pnn[:,:,j]-np.mean(pnn[:,:,j]),vmin=-0.2,vmax=0.2)
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
       wzn=fdt.computeVorticity(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],unn[:,:,j],vnn[:,:,j])
       
       fig, ax = plt.subplots(nrows=3,ncols=1)
       p1=ax[0].pcolor(Data.XX[:,:,j],Data.YY[:,:,j],WZ,vmin=-2,vmax=2)
       p2=ax[1].pcolor(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],WZHR,vmin=-2,vmax=2)
       p3=ax[2].pcolor(Data_HR.XX[:,:,j],Data_HR.YY[:,:,j],wzn,vmin=-2,vmax=2)
       ax[0].set_xlim([1, 8])
       ax[0].set_ylim([-2, 2])
       ax[1].set_xlim([1, 8])
       ax[1].set_ylim([-2, 2])
       ax[2].set_xlim([1, 8])
       ax[2].set_ylim([-2, 2])
       plt.savefig(directory+f'/wz_{j}.png')
       plt.close('all')
       
       xn = Data_HR.XX
       yn = Data_HR.YY
       tn = Data_HR.TT
       
       inputs = [Data_HR.XX,Data_HR.YY,Data_HR.TT]
       un,vn,pn = problem.domain.neural_net(state,'nnet')(*inputs)
       wzn = np.zeros(un.shape)
       for j in range(wzn.shape[2]):
          wzn[:,:,j] = fdt.computeVorticity(xn[:,:,j],yn[:,:,j],un[:,:,j],vn[:,:,j])
       
       # save output data
       mdic = {'x_sol':xn, 'y_sol':yn, 't_sol':tn, 'u_sol':un, 'v_sol':vn, 'wz_sol':wzn, 'p_sol':pn}
       
       sc.io.savemat(directory+'/solutionData.mat',mdic)


odil.setup_outdir(args)

# setup problem
problem,state=mkp.make_problem(args,Data)


# add a history function for error between true field and n-n
def history_func(problem, state, epoch, history, cbinfo):
    domain = problem.domain
    inputs = [Data_HR.XX,Data_HR.YY,Data_HR.TT]
    unn,vnn,pnn = problem.domain.neural_net(state,'nnet')(*inputs)
    #unn, = problem.domain.neural_net(state,'u_net')(*inputs)
    #vnn, = problem.domain.neural_net(state,'v_net')(*inputs)
    #pnn, = problem.domain.neural_net(state,'p_net')(*inputs)
    pnn=np.array(pnn)
    for j in range(pnn.shape[2]):
      pnn[:,:,j] = pnn[:,:,j] +  - np.mean(pnn[:,:,j]) + np.mean(Data_HR.PP[:,:,j])
    
    eUtrue= np.sqrt(np.mean((unn - Data_HR.UU)**2)) 
    eVtrue= np.sqrt(np.mean((vnn - Data_HR.VV)**2)) 
    ePtrue= np.sqrt(np.mean((pnn - Data_HR.PP)**2)) 
    
    extra=problem.extra
    
    inputs = [extra.Xn,extra.Yn,extra.Tn]
    unn,vnn,pnn = problem.domain.neural_net(state,'nnet')(*inputs)
    #unn, = problem.domain.neural_net(state,'u_net')(*inputs)
    #vnn, = problem.domain.neural_net(state,'v_net')(*inputs)
    #pnn, = problem.domain.neural_net(state,'p_net')(*inputs)
    
    UDatLoss = np.mean(np.abs(unn - extra.Un))
    VDatLoss = np.mean(np.abs(vnn - extra.Vn))
    
    UDatLoss_RMS = np.sqrt(np.mean((unn - extra.Un)**2))
    VDatLoss_RMS = np.sqrt(np.mean((vnn - extra.Vn)**2))
    
    inputs = [extra.Xv,extra.Yv,extra.Tv]
    unn,vnn,pnn = problem.domain.neural_net(state,'nnet')(*inputs)
    #unn, = problem.domain.neural_net(state,'u_net')(*inputs)
    #vnn, = problem.domain.neural_net(state,'v_net')(*inputs)
    #pnn, = problem.domain.neural_net(state,'p_net')(*inputs)
    
    UValLoss = np.mean(np.abs(unn - extra.Uv))
    VValLoss = np.mean(np.abs(vnn - extra.Vv))
    
    UValLoss_RMS = np.sqrt(np.mean((unn - extra.Uv)**2))
    VValLoss_RMS = np.sqrt(np.mean((vnn - extra.Vv)**2))
    
    
    # Add current parameters to `train.csv`.
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
bfgs_m=20
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




xn = Data_HR.XX
yn = Data_HR.YY
tn = Data_HR.TT

inputs = [Data_HR.XX,Data_HR.YY,Data_HR.TT]
un,vn,pn = problem.domain.neural_net(state,'nnet')(*inputs)
wzn = np.zeros(un.shape)
for j in range(wzn.shape[2]):
     wzn[:,:,j] = fdt.computeVorticity(xn[:,:,j],yn[:,:,j],un[:,:,j],vn[:,:,j])

# save output data
mdic = {'x_sol':xn, 'y_sol':yn, 't_sol':tn, 'u_sol':un, 'v_sol':vn, 'wz_sol':wzn, 'p_sol':pn, 'x_dat':Data.XX, 'y_dat':Data.YY, 't_dat':Data.TT, 'u_dat':Data.UU, 'v_dat':Data.VV, 'wz_dat':Data.WZ, 'p_dat':Data.PP, 'x_hr':Data_HR.XX, 'y_hr':Data_HR.YY, 't_hr':Data_HR.TT, 'u_hr':Data_HR.UU, 'v_hr':Data_HR.VV, 'p_hr':Data_HR.PP}

sc.io.savemat('solutionData.mat',mdic)





# save neural network data
NN_arch = []
for j in range(args.nHL):
  NN_arch.append(args.nNHL)

mdic2= {'nnarch':NN_arch,'weights':state.fields['nnet'].weights,'biases':state.fields['nnet'].biases}


sc.io.savemat('nnData.mat',mdic2)
