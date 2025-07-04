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
   pn=ctx.field('pn')
   
   # downsample to data size:
   nx,ny,nt=domain.size(loc='nnn')
   nx1,ny1,nt1=extra.Un.shape
   
   u2 = un[(extra.pad1[0]):(extra.pad1[0]+(nx1-1)*extra.usf[0]+1):extra.usf[0],(extra.pad1[1]):(extra.pad1[1]+(ny1-1)*extra.usf[1]+1):extra.usf[1],(extra.pad1[2]):(extra.pad1[2]+(nt1-1)*extra.usf[2]+1):extra.usf[2]]
   v2 = vn[(extra.pad1[0]):(extra.pad1[0]+(nx1-1)*extra.usf[0]+1):extra.usf[0],(extra.pad1[1]):(extra.pad1[1]+(ny1-1)*extra.usf[1]+1):extra.usf[1],(extra.pad1[2]):(extra.pad1[2]+(nt1-1)*extra.usf[2]+1):extra.usf[2]]
   
   print(extra.Mask.shape)
   print(extra.Un.shape)
   print(u2.shape)
   
   res+=[('measurement: ux', extra.Weights[0]*extra.Mask*extra.MW*(u2 - extra.Un))]
   res+=[('measurement: uy', extra.Weights[0]*extra.Mask*extra.MW*(v2 - extra.Vn))] 
   
   
   # get velocity and pressure gradients
   dx, dy, dt = ctx.step('x', 'y', 't')
   def stencil_roll_n(q):
        return [
            mod.roll(q, shift=np.negative(s), axis=(0, 1, 2))
            for s in [(0,0,0),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(-1,-1,0),(1,-1,0),(-1,1,0),(1,1,0),(0,0,1),(-1,0,1),(1,0,1),(0,-1,1),(0,1,1),(-1,-1,1),(1,-1,1),(-1,1,1),(1,1,1)]
        ]
   def gradients_n(q):
       q_st = stencil_roll_n(q)
       qp,qwp,qep,qsp,qnp,qswp,qsep,qnwp,qnep,qf,qwf,qef,qsf,qnf,qswf,qsef,qnwf,qnef = q_st
       q_x = ((qep+qef)-(qwp+qwf))/(4*dx)
       q_y = ((qnp+qnf)-(qsp+qsf))/(4*dy)
       q_t = (qf-qp)/dt
       q_xx=((qep+qef)-2*(qp+qf)+(qwp+qwf))/(2*dx**2)
       q_yy=((qnp+qnf)-2*(qp+qf)+(qsp+qsf))/(2*dy**2)
       q_t = q_t[2:-2,2:-2,0:-1]
       q_x = q_x[2:-2,2:-2,0:-1]
       q_y = q_y[2:-2,2:-2,0:-1]
       q_xx= q_xx[2:-2,2:-2,0:-1]
       q_yy= q_yy[2:-2,2:-2,0:-1]    
       return q_t,q_x,q_y,q_xx,q_yy
   
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
   p_t,p_x,p_y,p_xx,p_yy = gradients_n(pn)    
   
   uu_x,vu_y,q_t = upwindAdvection(un,un,vn)
   uv_x,vv_y,q_t = upwindAdvection(vn,un,vn)
   
   Re = 100
   un1 = un[2:-2,2:-2,0:-1]
   vn1 = vn[2:-2,2:-2,0:-1]
      
   res+=[('x-momentum',extra.Weights[1]*(u_t+uu_x+vu_y+p_x-1/Re*(u_xx+u_yy)))]
   res+=[('y-momentum',extra.Weights[1]*(v_t+uv_x+vv_y+p_y-1/Re*(v_xx+v_yy)))]
   res+=[('continuity',extra.Weights[1]*(u_x+v_y))]
     
   
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
     L4u = gradients_L4(un)
     L4v = gradients_L4(vn)
     L4p = gradients_L4(pn)
     ptt = gradient_t2(pn)   
     Su = L4u
     Sv = L4v
     
     
     res+=[('u-smoothing',extra.Weights[2]*Su)]
     res+=[('v-smoothing',extra.Weights[2]*Sv)]      
     
     L4p = gradients_L4(pn)
     ptt = gradient_t2(pn) 
     Sp = L4p
     Spt = ptt
     
     res+=[('p-smoothing',extra.Weights[2]*Sp)]   
     res+=[('pt-smoothing',extra.Weights[2]*Spt)]
     
   return res
