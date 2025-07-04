import numpy as np
import scipy as sc
import matplotlib
from matplotlib import pyplot as plt

def operator(ctx):
   res=[]
   extra = ctx.extra
   mod = ctx.mod
   tf = mod.tf
   
   # measurement data
   #inputs_1 = [mod.constant(extra.Xn), mod.constant(extra.Yn), mod.constant(extra.Tn)]
   #nndat = ctx.neural_net('u_net')(*inputs_1)
   #udat,vdat,fxdat,fydat = ctx.neural_net('nnet')(*inputs_1)
   #udat = nndat[0]
   #vdat = nndat[0]
   #print(udat.shape)
   #udat, = ctx.neural_net('u_net')(*inputs_1)
   #vdat, = ctx.neural_net('v_net')(*inputs_1)
   

   
   #res+=[('measurement: ux', extra.Weights[0]*mod.sqrt(mod.abs(udat - extra.Un)))]
   #res+=[('measurement: uy', extra.Weights[0]*mod.sqrt(mod.abs(vdat - extra.Vn)))]
   
   
   # inner points
   # calculate physics at both data and physics nodes?
   print(extra.Xn.shape)
   print(extra.x_inner.shape)
   x2 = np.hstack((extra.Xn,extra.x_inner))
   y2 = np.hstack((extra.Yn,extra.y_inner))
   t2 = np.hstack((extra.Tn,extra.t_inner))
   print(x2.shape)
   inputs = [mod.constant(x2), mod.constant(y2), mod.constant(t2)]
   u,v,fx,fy = ctx.neural_net('nnet')(*inputs)
   #u, = ctx.neural_net('u_net')(*inputs)
   #v, = ctx.neural_net('v_net')(*inputs)
   #fx, = ctx.neural_net('fx_net')(*inputs)
   #fy, = ctx.neural_net('fy_net')(*inputs)
   
   def grad(f, *deriv):
        for i in range(len(deriv)):
            for _ in range(deriv[i]):
                f = tf.gradients(f, inputs[i])[0]
        return f

   u_t = grad(u, 0, 0, 1)
   u_x = grad(u, 1, 0, 0)
   u_y = grad(u, 0, 1, 0)
   u_xx= grad(u, 2, 0, 0)
   u_yy= grad(u, 0, 2, 0)
   
   v_t = grad(v, 0, 0, 1)
   v_x = grad(v, 1, 0, 0)
   v_y = grad(v, 0, 1, 0)
   v_xx= grad(v, 2, 0, 0)
   v_yy= grad(v, 0, 2, 0)
   
   #fx_t = grad(p, 0, 0, 1)
   #fx_x = grad(p, 1, 0, 0)
   #fx_y = grad(p, 0, 1, 0)
   #p_xx= grad(p, 2, 0, 0)
   #p_yy= grad(p, 0, 2, 0)
   
   Re = extra.Re
   
   res+=[('measurement: ux', extra.Weights[0]*(u[0:extra.Un.shape[0]] - extra.Un))]
   res+=[('measurement: uy', extra.Weights[0]*(v[0:extra.Un.shape[0]] - extra.Vn))]
   
   res+=[('x-momentum',extra.Weights[1]*(u_t+u*u_x+v*u_y-fx-1/Re*(u_xx+u_yy)))]
   res+=[('y-momentum',extra.Weights[1]*(v_t+u*v_x+v*v_y-fy-1/Re*(v_xx+v_yy)))]
   
   
   return res
