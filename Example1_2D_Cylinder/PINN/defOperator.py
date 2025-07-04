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
   inputs_1 = [mod.constant(extra.Xn), mod.constant(extra.Yn), mod.constant(extra.Tn)]
   udat,vdat,pdat=ctx.neural_net('nnet')(*inputs_1)
   #udat, = ctx.neural_net('u_net')(*inputs_1)
   #vdat, = ctx.neural_net('v_net')(*inputs_1)
   
   res+=[('measurement: ux', extra.Weights[0]*(udat - extra.Un))]
   res+=[('measurement: uy', extra.Weights[0]*(vdat - extra.Vn))]
   
   
   # inner points
   inputs = [mod.constant(extra.x_inner), mod.constant(extra.y_inner), mod.constant(extra.t_inner)]
   u,v,p = ctx.neural_net('nnet')(*inputs)
   #u, = ctx.neural_net('u_net')(*inputs)
   #v, = ctx.neural_net('v_net')(*inputs)
   #p, = ctx.neural_net('p_net')(*inputs)
   
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
   
   p_t = grad(p, 0, 0, 1)
   p_x = grad(p, 1, 0, 0)
   p_y = grad(p, 0, 1, 0)
   p_xx= grad(p, 2, 0, 0)
   p_yy= grad(p, 0, 2, 0)
   
   Re = 100
   
   res+=[('x-momentum',extra.Weights[1]*(u_t+u*u_x+v*u_y+p_x-1/Re*(u_xx+u_yy)))]
   res+=[('y-momentum',extra.Weights[1]*(v_t+u*v_x+v*v_y+p_y-1/Re*(v_xx+v_yy)))]
   res+=[('continuity',extra.Weights[1]*(u_x+v_y))]
   
   print(u.shape)
   
   return res
