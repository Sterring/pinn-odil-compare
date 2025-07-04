import numpy as np
import scipy as sc
import matplotlib
from matplotlib import pyplot as plt



def operator(ctx):
   domain = ctx.domain
   extra = ctx.extra
   mod = ctx.mod
      
   res = []
   
   # loss function based on measured data:
   un=ctx.field('un')
   vn=ctx.field('vn')
   fxn=ctx.field('fxn')
   fyn=ctx.field('fyn')
   
   # downsample to data size:
   nx,ny,nt=domain.size(loc='nnn') 
   nx1,ny1,nt1=extra.Un.shape
   
   print(nx)
   print(extra.pad1)
   print(extra.usf)
   
   u2 = un[(extra.pad1[0]):(extra.pad1[0]+(nx1-1)*extra.usf[0]+1):extra.usf[0],(extra.pad1[1]):(extra.pad1[1]+(ny1-1)*extra.usf[1]+1):extra.usf[1],(extra.pad1[2]):(extra.pad1[2]+(nt1-1)*extra.usf[2]+1):extra.usf[2]]
   v2 = vn[(extra.pad1[0]):(extra.pad1[0]+(nx1-1)*extra.usf[0]+1):extra.usf[0],(extra.pad1[1]):(extra.pad1[1]+(ny1-1)*extra.usf[1]+1):extra.usf[1],(extra.pad1[2]):(extra.pad1[2]+(nt1-1)*extra.usf[2]+1):extra.usf[2]]
   
   print(u2)
   print(extra.Un.shape)
   res+=[('measurement-ux', extra.Weights[0]*extra.Mask*extra.MW*(u2 - extra.Un))]
   res+=[('measurement-uy', extra.Weights[0]*extra.Mask*extra.MW*(v2 - extra.Vn))]
   
   # get velocity and pressure gradients
   dx, dy, dt = ctx.step('x', 'y', 't')
   def stencil_roll_n(q):
        return [
            mod.roll(q, shift=np.negative(s), axis=(0, 1, 2))
            for s in [(0,0,0),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,1),(-1,0,1),(1,0,1),(0,-1,1),(0,1,1)]
            #            cp      wp      ep       sp      np       cf        wf      ef      sf       nf 
        ]
   def gradients_n(q):
       q_st = stencil_roll_n(q)
       qcp, qwp, qep, qsp, qnp, qcf, qwf, qef, qsf, qnf  = q_st
       
       q_t= (qcf-qcp)/dt
       
       q_x = ((qep+qef) - (qwp+qwf))/(4*dx)
       q_y = ((qnp+qnf) - (qsp+qsf))/(4*dy)
       
       q_xx= ((qep+qef)-2*(qcp+qcf)+(qwp+qwf))/(2*dx**2)
       q_yy= ((qnp+qnf)-2*(qcp+qcf)+(qsp+qsf))/(2*dy**2)
       
       q_t = q_t[2:-2,2:-2,0:-1]
       q_x = q_x[2:-2,2:-2,0:-1]
       q_y = q_y[2:-2,2:-2,0:-1]
       q_xx= q_xx[2:-2,2:-2,0:-1]
       q_yy= q_yy[2:-2,2:-2,0:-1]       
       return q_t, q_x, q_y, q_xx, q_yy
   
   def stencil_roll_2(q):
        return [
            mod.roll(q, shift=np.negative(s), axis=(0, 1, 2))
            for s in [(0,0,0),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(-2,0,0),(2,0,0),(0,-2,0),(0,2,0),(0,0,1),(-1,0,1),(1,0,1),(0,-1,1),(0,1,1),(-2,0,1),(2,0,1),(0,-2,1),(0,2,1),]
            #            cp      wp      ep       sp      np       wwp      eep     ssp     nnp      cf      wf      ef      sf       nf        wwf    eef      ssf      nnf
        ]
   
   def upwindAdvection(q,u,v):
        qcp, qwp, qep, qsp, qnp, qwwp, qeep, qssp, qnnp, qcf, qwf, qef, qsf, qnf, qwwf, qeef, qssf, qnnf = stencil_roll_2(q)
        
        q_xp = (-(qeef+qeep) + 4*(qef+qep) - 3 *(qcf+qcp))/(4*dx)
        q_xn = ( (qwwf+qwwp) - 4*(qwf+qwp) + 3 *(qcf+qcp))/(4*dx)
        q_yp = (-(qnnf+qnnp) + 4*(qnf+qnp) - 3 *(qcf+qcp))/(4*dy)
        q_yn = ( (qssf+qssp) - 4*(qsf+qsp) + 3 *(qcf+qcp))/(4*dy)

        # get sign of velocity for upwinding
        up = mod.tf.math.maximum(u,0)
        un = mod.tf.math.minimum(u,0)
        vp = mod.tf.math.maximum(v,0)
        vn = mod.tf.math.minimum(v,0)
        
        uq_x = up*q_xn + un*q_xp
        vq_y = vp*q_yn + vn*q_yp
        
        q_t= (qcf-qcp)/dt
        
        uq_x = uq_x[2:-2,2:-2,0:-1]
        vq_y = vq_y[2:-2,2:-2,0:-1]
        
        q_t  = q_t[2:-2,2:-2,0:-1]
        return uq_x,vq_y,q_t
        
   u_t,u_x,u_y,u_xx,u_yy = gradients_n(un)     
   v_t,v_x,v_y,v_xx,v_yy = gradients_n(vn)       
   
   uu_x,vu_y,q_t = upwindAdvection(un,un,vn)
   uv_x,vv_y,q_t = upwindAdvection(vn,un,vn)
   
   ufx_x,vfx_y,fx_t = upwindAdvection(fxn,un,vn)
   ufy_x,vfy_y,fy_t = upwindAdvection(fyn,un,vn)
   
   # material derivatives of forcings
   Dfx = fx_t + ufx_x + vfx_y
   Dfy = fy_t + ufy_x + vfy_y
   
   Re = extra.Re
   un1 = un[2:-2,2:-2,0:-1]
   vn1 = vn[2:-2,2:-2,0:-1]
   fxn1= fxn[2:-2,2:-2,0:-1]
   fyn1= fyn[2:-2,2:-2,0:-1]
   
   res+=[('x-momentum',extra.Weights[1]*(u_t+uu_x+vu_y-fxn1-1/Re*(u_xx+u_yy)))]
   res+=[('y-momentum',extra.Weights[1]*(v_t+uv_x+vv_y-fyn1-1/Re*(v_xx+v_yy)))]
   
   def stencil_roll_3(q):
        return [
            mod.roll(q, shift=np.negative(s), axis=(0, 1, 2))
            for s in [(0,0,0),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(-2,0,0),(2,0,0),(0,-2,0),(0,2,0)]
            #            c      w         e       s       n       ww        ee      ss      nn     
        ]
   def gradients_L4(q):
       q_st = stencil_roll_3(q)
       qc, qw, qe, qs, qn, qww, qee, qss, qnn = q_st
       q_x4 = (qww-4*qw+6*qc-4*qe+qee)/(dx**4)
       q_y4 = (qss-4*qs+6*qc-4*qn+qnn)/(dy**4)
       L4q = q_x4+q_y4
       L4q = L4q[2:-2,2:-2,:]
       return L4q
   def stencil_roll_t(q):
       return [
            mod.roll(q, shift=np.negative(s), axis=(0, 1, 2))
            for s in [(0,0,0),(0,0,-1),(0,0,1)]
       ]
   def gradient_t2(q):
       q_st = stencil_roll_t(q)
       qc, qp, qf = q_st
       q_tt = (qf-2*qc+qp)/(dt**2)
       q_tt = q_tt[:,:,1:-1]
       return q_tt


   if extra.Weights[2]>0:
     dU = u_x+v_y
     dU_t,dU_x,dU_y,dU_xx,dU_yy = gradients_n(dU)
     L2dU = dU_xx+dU_yy
     SDdU = L2dU
     res+=[('smoothing-dU: L2',extra.Weights[2]*SDdU)]

   
   if extra.Weights[3]>0:
     Dfx_t,Dfx_x,Dfx_y,Dfx_xx,Dfx_yy = gradients_n(Dfx) 
     Dfy_t,Dfy_x,Dfy_y,Dfy_xx,Dfy_yy = gradients_n(Dfy) 
     L2Dfx = Dfx_xx+Dfx_yy
     L2Dfy = Dfy_xx+Dfy_yy
     SDfx = 1e-4*L2Dfx
     SDfy = 1e-4*L2Dfy
     res+=[('smoothing-fx: L2',extra.Weights[3]*SDfx)]
     res+=[('smoothing-fy: L2',extra.Weights[3]*SDfy)]
     
   
   
   return res
