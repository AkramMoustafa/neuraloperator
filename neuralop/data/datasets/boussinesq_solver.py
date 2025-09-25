import numpy as np
import math
import configparser
import matplotlib.pyplot as plt
PI = 3.14159265358979323846264338327950288419716939937510582097494459230781
L = 2*PI
A = 3.0
r"""
August 13, 2025
 * Xiaoming Zheng
 * Akram Moustafa
 * Department of Mathematics
 * Central Michigan University
 * Mount Pleasant, MI 48859
 *
   Solve Perturbation Of Anistropic Boussinest Equation
   from the paper
Adhikari, D., Ben Said, O., Pandey, U.R., Wu Jiahong Stability and large-time behavior for the 2D Boussineq system with horizontal dissipation and vertical thermal diffusion. Nonlinear Differ. Equ. Appl. 29, 42 (2022). https://doi.org/10.1007/s00030-022-00773-4

Purpose: finding solution
   u = (u1 u2) and theta satisfying Equation: (1.10)

   D_tt u     - (eta D_22 + nu D_11) D_t u     + nu*eta*D_11 D_22 u     + D_11 Laplace^{-1} u     = N5
   D_tt theta - (eta D_22 + nu D_11) D_t theta + nu*eta*D_11 D_22 theta + D_11 Laplace^{-1} theta = N6

where
   N5 = - (D_t - η D_22) P(u·∇u) + ∇^⊥ D_1 Laplace⁻¹(u·∇θ)
   N6 = - (D_t - ν D_11)(u·∇θ) + u·∇u₂ - D_2 Laplace⁻¹ ∇(u·∇u)

where P is Leray's projection, that is,
   P(u) = u - \nabla \Laplace^{-1} (\nabla\cdot u)

Method: convert to first order differential system on t: introducing new variables
   v1  = D_t u1
   v2  = D_t u2
   ps = D_t theta

Two files are need to run this file which include an input vorticity file and configuration file. Please refer to this link to know how to generate the input vorticity.
https://github.com/neuraloperator/pino-closure-models/blob/master/kf/solver/random_fields.py
"""

def g1_func(t, x, y,ConvTest, nu):

    if(ConvTest):
        # //w=  -cos(t)*2*a*cos(2*a*x)*pow(sin(a*y),2)  -cos(t)*pow(sin(a*x),2)*2*a*cos(2*a*y)
        # //theta= cos(t)*pow(sin(a*x),2)*sin(2*a*y)

        u1   = np.cos(t)*pow(np.sin(A*x),2)*np.sin(2*A*y)
        u2   =-np.cos(t)*np.sin(2*A*x)*pow(np.sin(A*y),2)
        Dtw  = np.sin(t)*2*A*np.cos(2*A*x)*pow(np.sin(A*y),2) + np.sin(t)*pow(np.sin(A*x),2)*2*A*np.cos(2*A*y)
        D1w  = np.cos(t)*4*A*A*np.sin(2*A*x)*pow(np.sin(A*y),2)-np.cos(t)*np.sin(2*A*x)*2*A*A*np.cos(2*A*y)
        D2w  =-np.cos(t)*2*A*A*np.cos(2*A*x)*np.sin(2*A*y) + np.cos(t)*pow(np.sin(A*x),2)*4*A*A*np.sin(2*A*y)
        D11w = np.cos(t)*8*A*A*A*np.cos(2*A*x)*pow(np.sin(A*y),2) -np.cos(t)*np.cos(2*A*x)*4*A*A*A*np.cos(2*A*y)
        D1th = np.cos(t)*A*np.sin(2*A*x)*np.sin(2*A*y)
        tmp = Dtw - nu*D11w

        # return Dtw - nu*D11w -D1th
        return Dtw + u1*D1w + u2*D2w - D1th - nu*D11w
    else:
      return np.zeros_like(x, dtype=float)

def f3_func( t,  x,  y, ConvTest):

    if( ConvTest):

        u1   = np.cos(t)*pow(np.sin(A*x),2)*np.sin(2*A*y)
        u2   =-np.cos(t)*np.sin(2*A*x)*pow(np.sin(A*y),2)
        th   = u1
        Dtth =-np.sin(t)*pow(np.sin(A*x),2)*np.sin(2*A*y)
        D1th = np.cos(t)*A*np.sin(2*A*x)*np.sin(2*A*y)
        D2th = np.cos(t)*pow(np.sin(A*x),2)*2*A*np.cos(2*A*y)
        D22th=-np.cos(t)*pow(np.sin(A*x),2)*4*A*A*np.sin(2*A*y)

        # if(t>0 and x>0):
        #   # print(f" t,  x,  y: {t},  {x},  {y}, {Dtth:12.5e}")

        #   # input("continue?")
        return Dtth + u1*D1th + u2*D2th + u2- eta*D22th

    else:
        return np.zeros_like(x, dtype=float)

def f2_func( t,  x,  y, a, ConvTest):
    if( ConvTest ):

        u1   = np.cos(t) * (np.sin(A*x)**2) * np.sin(2*A*y)
        u2   = -np.cos(t) * np.sin(2*A*x) * (np.sin(A*y)**2)
        th   = u1
        p    = np.cos(t) * np.sin(2*A*x) * np.sin(2*A*y)

        Dtu2 = np.sin(t) * np.sin(2*A*x) * (np.sin(A*y)**2)
        D1u2 = -np.cos(t) * 2*A * np.cos(2*A*x) * (np.sin(A*y)**2)
        D2u2 = -np.cos(t) * np.sin(2*A*x) * A * np.sin(2*A*y)
        D11u2 = np.cos(t) * 4*A**2 * np.sin(2*A*x) * (np.sin(A*y)**2)
        D2p   = np.cos(t) * np.sin(2*A*x) * 2*A * np.cos(2*A*y)

        return Dtu2 + u1*D1u2 + u2*D2u2 + D2p - nu*D11u2 - th
    else:
        return np.zeros_like(x, dtype=float)

def f1_func( t,  x,  y, ConvTest):

    if( ConvTest ):
        u1   = np.cos(t) * (np.sin(A*x)**2) * np.sin(2*A*y)
        u2   = -np.cos(t) * np.sin(2*A*x) * (np.sin(A*y)**2)
        p    = np.cos(t) * np.sin(2*A*x) * np.sin(2*A*y)

        Dtu1 = -np.sin(t) * (np.sin(A*x)**2) * np.sin(2*A*y)
        D1u1 = np.cos(t) * A * np.sin(2*A*x) * np.sin(2*A*y)
        D2u1 = np.cos(t) * (np.sin(A*x)**2) * 2*A * np.cos(2*A*y)
        D11u1 = np.cos(t) * 2*A**2 * np.cos(2*A*x) * np.sin(2*A*y)
        D1p   = np.cos(t) * 2*A * np.cos(2*A*x) * np.sin(2*A*y)

        return Dtu1 + u1*D1u1 + u2*D2u1 + D1p - nu*D11u1
    else:
        return np.zeros_like(x, dtype=float)

def winitFunc(t, x, y):
    if ConvTest:
        return -np.cos(t) * 2*A*np.cos(2*A*x) * np.sin(A*y)**2 \
               -np.cos(t) * (np.sin(A*x)**2) * 2*A*np.cos(2*A*y)
    else:
        return -WuEpsi*np.cos(t) * 2*A*np.cos(2*A*x) * np.sin(A*y)**2 \
               -WuEpsi*np.cos(t) * (np.sin(A*x)**2) * 2*A*np.cos(2*A*y)

def prinitFunc( t,  x,  y):
  if(ConvTest):
     return np.cos(t)*np.sin(2*A*x)*np.sin(2*A*y);
  else:
     return WuEpsi*np.cos(t)*pow(np.sin(A*x),2)*np.sin(2*A*y);

def thinitFunc( t,  x,  y):
# // initial value of theta
    if ConvTest:
        return np.cos(t) * (np.sin(A * x) ** 2) * np.sin(2 * A * y)
    else:
        nX = 4
        tmp0 = 4 * (1.0 + 1.0/4 + 1.0/9 + 1.0/16)
        tmp1 = tmp0 - np.sum([4.0/(i*i) * np.cos(i * x) for i in range(1, nX+1)], axis=0)
        tmp2 = tmp0 - np.sum([4.0/(i*i) * np.cos(i * y) for i in range(1, nX+1)], axis=0)
          #  return WuEpsi*math.cos(t)* (tmp1 * tmp2 - tmp0*tmp0)
        return WuEpsi * np.cos(t) * (tmp1 * tmp2 - tmp0 * tmp0)

def u1initFunc( t,  x,  y):
  if(ConvTest):
     return np.cos(t)*pow(np.sin(A*x),2)*np.sin(2*A*y)
  else:
     return WuEpsi*np.cos(t)*pow(np.sin(A*x),2)*np.sin(2*A*y)

def u2initFunc( t,  x,  y):
  if(ConvTest):
     return -np.cos(t)*np.sin(2*A*x)*pow(np.sin(A*y),2)
  else:
     return -WuEpsi*np.cos(t)*np.sin(2*A*x)*pow(np.sin(A*y),2)

def AfterNonlinear( u1, u2, w, th, time_spec):

    th = np.fft.fftn(th, s=(N0, N1),axes=(0, 1) ,norm="backward")
    u1 = np.fft.fftn(u1, s=(N0, N1), axes=(0, 1),norm="backward")
    u2 = np.fft.fftn(u2, s=(N0, N1), axes=(0, 1),norm="backward")
    w  = np.fft.fftn(w,  s=(N0, N1), axes=(0, 1),norm="backward")

    return u1, u2, w, th

def PreNonlinear(u1, u2, w, th,N3, N4 , time_spec):

    # Above all, convert input quantities to Physical space:
    u1 = np.fft.ifftn(u1, s=(N0, N1), axes=(0, 1), norm="backward").real
    u2 = np.fft.ifftn(u2, s=(N0, N1), axes=(0, 1), norm="backward").real
    th = np.fft.ifftn(th, s=(N0, N1), axes=(0, 1), norm="backward").real
    w  = np.fft.ifftn(w,  s=(N0, N1), axes=(0, 1), norm="backward").real

    # divide them by DIM because FFTW does not perform it:
    # initialize N5 and N6:
    N3[:] = 0.0 + 0.0j
    N4[:] = 0.0 + 0.0j

    return N3, N4,u1, u2, w, th
def Nonlinear3and4(u1, u2, w, th, tmp1,  tmp2,  tmp3, tmp4,   N3, N4, time_spec):

    # computer N3= (u\cdot\nabla)\theta
    # computer N4= (u\cdot\nabla) w
    # Denote z = hat (u\cdot nabla)\theta = hat( div(u*theta) ):
    #          = hat( D_1(u1*theta) + D_2(u2*theta) )

    # first we need u1*theta, u1*theta in Physical space
#    set u1 and u2 as exact values to check on p:
    tmp1[:] = (u1.real * th.real) + 0.0j   # u1 * theta
    tmp2[:] = (u2.real * th.real) + 0.0j   # u2 * theta
    tmp3[:] = (u1.real * w.real)  + 0.0j   # u1 * vort
    tmp4[:] = (u2.real * w.real)  + 0.0j   # u2 * vort

    tmp1 = np.fft.fftn(tmp1, s=(N0, N1), axes=(0, 1), norm="backward")
    tmp2 = np.fft.fftn(tmp2, s=(N0, N1), axes=(0, 1), norm="backward")
    tmp3 = np.fft.fftn(tmp3, s=(N0, N1), axes=(0, 1), norm="backward")
    tmp4 = np.fft.fftn(tmp4, s=(N0, N1), axes=(0, 1), norm="backward")

    kx = wn[:, None]
    ky = wn[None, :]
    # Third compute n3 and n4
    N3[:, :] =  1j*(kx * tmp1 + ky * tmp2)
    N4[:, :] =  1j*(kx * tmp3 + ky * tmp4)

    return tmp1, tmp2, tmp3, tmp4, N3, N4


def Read_VorTemp():
    data = np.loadtxt("/content/drive/MyDrive/initial_vorticity.txt")

    if data.size != N0 * N1:
        raise ValueError(f"File has {data.size} values but expected {N0*N1}")

    # flatten to 1D and cast to complex128
    w = data.astype(np.complex128)

    return w

def LerayProjectionFourier(w1,  w2, k1, k2, ksq):
    # perform Leray projection in the Fourier space
    # input: w=(w1, w2) in Fourier mode
    # output: (out1 , out2) = Fourier mode of P(w)
    # hat(P(w)) = hat(w) - hat( grad InverseLaplace div (w) )
    #          = hat(w) if k=0
    #          = 1/|k|^2 [ k2^2  -k1*k2 ] [hat(w1)] when k\ne 0
    #                    [-k1*k2   k1^2 ] [hat(w2)]

    if( ksq > 1e-2 ) :
         tmp1 =  (k2*w1.real - k1*w2.real)*k2/ksq;
         tmp2 =  (k2*w1.imag - k1*w2.imag)*k2/ksq;
         tmp3 =  (k1*w2.real - k2*w1.real)*k1/ksq;
         tmp4 =  (k1*w2.imag - k2*w1.imag)*k1/ksq;
         w1 = tmp1 + 1j * tmp2
         w2 = tmp3 + 1j * tmp4

r"""
k2 is F(tn+dt, u_n+dt*k1) of ODE u_t=F(t,u)
"""
def RHS(k_w, k_th, u1, u2, th, w,
        g1tmp, f3tmp, t,
        wn, N3,N4,tmp1, tmp2, tmp3, tmp4):
    # print("ttttttttttttttttttttttttttttttttttttttttttttttttttttt",t)
    # without diffusion
    k_w, k_th = RHS_BE(k_w, k_th, u1, u2, th, w, g1tmp, f3tmp, t, wn, N3, N4,tmp1, tmp2, tmp3, tmp4)

    k1 = wn[:, None]   # shape (N0, 1)
    k2 = wn[None, :]   # shape (1, N1)
    k1sq = k1**2
    k2sq = k2**2

    # diffusion added hgere
    k_w -= nu * k1sq * w
    k_th -= eta * k2sq * th

    return k_w, k_th


def RHS_k1_w_th(u1,u2,th, w, k1_w,  k1_th, dt, t, g1tmp, f3tmp, N3,N4,tmp1, tmp2, tmp3, tmp4):
    # Purpose: find k1_w =-\nabla\cdot(u w) + \nabla_x theta + g1
    # g1 = \nabla\times (f1,f2)
    # k1_th=-\nabla\cdot(u \theta) -u2 + f3
    # work in Fourier space

    # g1tmp is external term of equation of w_t
    # f3tmp is external term of equation of theta_t
    x = (np.arange(N0) * L / N0)[:, None]
    y = (np.arange(N1) * L / N1)[None, :]
    g1tmp = np.array(g1_func(t, x, y, ConvTest, nu), dtype=np.complex128)
    f3tmp = np.array(f3_func(t, x, y, ConvTest), dtype=np.complex128)

    g1tmp = np.fft.fftn(g1tmp, s=(N0, N1), axes=(0, 1), norm="backward")
    f3tmp = np.fft.fftn(f3tmp, s=(N0, N1), axes=(0, 1), norm="backward")

    # k1 is F(t_n, u_n) of ODE u_t=F(t,u) without diffusion terms:
    k1_w, k1_th = RHS_BE(k1_w, k1_th,u1, u2, th, w, g1tmp, f3tmp, t,wn, N3, N4,tmp1, tmp2, tmp3, tmp4)
    return k1_w, k1_th

def RHS_BE(k_w, k_th, u1, u2, th, w,
           g1tmp, f3tmp, t,
           wn, N3,N4,tmp1, tmp2, tmp3, tmp4):
    # print("RHS_BE ttttttttttttttttttttttttttttt:",t)
    # compute RHS of ODE in Fourier space without diffusion terms
    # input: in_* for *=u1, u2, th (theta), pr (pressure)
    # output: RHS k_* for *=u1, u2, th (theta), pr (pressure)
    # k is F(y,t) in ODE dy/dt=F(y,t) without diffusion
    # both input and output are in Fourier mod
    # The following arrays denote temporary ones in computing N5 and N6:

    # Nonlinear terms N11, N12, N2, N3:
    #  Prepare for nonlinear calculation:
    N3, N4, u1_phys, u2_phys, w_phys, th_phys = PreNonlinear(u1, u2, w, th, N3,N4, t)

    #  Step 1: Collect terms of (u\cdot\nabla)u to N11, N12 and N2:
    tmp1, tmp2, tmp3,tmp4 , N3,N4 =  Nonlinear3and4(u1_phys, u2_phys, w_phys, th_phys, tmp1, tmp2, tmp3, tmp4, N3,N4, t)
    k1 = wn[:, None]        # shape (N0, 1)
    k2 = wn[None, :]        # shape (1, N1)
    # Next, add linear terms:
    # rhs of w: dw/dt = g1 - N4 - i*k1*theta

    k_w[:, :] = g1tmp -N4 + 1j * k1 * th
    k_th[:, :] = f3tmp - N3 - u2

    return k_w, k_th

def do_IMEX(k1_w, k1_th, u1, u2, th, w, g1tmp, f3tmp, t, wn, N3,N4,tmp1, tmp2, tmp3, tmp4):

    # BE method on diffusion terms, explicit on other terms
    # u1n_in - in_u1 )/dt = f
    # has 4 unknowns to solve: u1, u2, th (theta), pr (pressure)
    # work in Fourier space
    # (f1tmp, f2tmp) is external term of equation of u_t
    # f3tmp is external term of equation of theta_t
    # x = (np.arange(N0) * L / N0)[:, None]   # shape (N0, 1)
    # y = (np.arange(N1) * L / N1)[None, :]   # shape (1, N1)
    # g1tmp[:, :] = g1_func(t, x, y, ConvTest, nu)
    # f3tmp[:, :] = f3_func(t, x, y, ConvTest)

    # # Forward FFT
    # g1tmp = np.fft.fftn(g1tmp,s=(N0, N1), axes=(0, 1), norm="backward")
    # f3tmp = np.fft.fftn(f3tmp,s=(N0, N1), axes=(0, 1), norm="backward")
    # # first, compute new u1, u2, and theta
    # k1_w, k1_th = RHS_BE(k1_w, k1_th, u1, u2, th, w, g1tmp, f3tmp, t, wn, N3,N4,tmp1, tmp2, tmp3, tmp4)

    # #  Diffusion terms
    # k1 = wn[:, None]
    # k2 = wn[None, :]
    # k1sq = k1**2
    # k2sq = k2**2
    # print(k1)
    # denom_w  = 1.0 + dt *nu* k1sq
    # denom_th = 1.0 + dt *eta* k2sq

    # w  = (w  + dt * k1_w)  / denom_w
    # th = (th + dt * k1_th) / denom_th

    # return w, th
    u1 = u1.ravel()
    u2 = u2.ravel()
    th = th.ravel()
    w  = w.ravel()
    g1tmp = g1tmp.ravel()
    f3tmp = f3tmp.ravel()
    k1_w = k1_w.ravel()
    k1_th = k1_th.ravel()

    for i in range(N0):
       for j in range(N1):
          jj = i*N1+j;
          x=L*(i)/N0;
          y=L*j/N1;
          g1tmp[jj] = g1_func(t, x,y,ConvTest,nu=1)
          f3tmp[jj]= f3_func(t, x,y,ConvTest)
    g1tmp2d = g1tmp.reshape(N0, N1)
    f3tmp2d = f3tmp.reshape(N0, N1)
    g1tmp_fft = np.fft.fftn(g1tmp2d, s=(N0, N1), norm="backward").ravel()
    f3tmp_fft = np.fft.fftn(f3tmp2d, s=(N0, N1), norm="backward").ravel()

    # first, compute new u1, u2, and theta

    k1_w, k1_th = RHS_BE(k1_w.reshape(N0, N1),k1_th.reshape(N0, N1), u1.reshape(N0, N1),u2.reshape(N0, N1),th.reshape(N0, N1),w.reshape(N0, N1),
    g1tmp_fft.reshape(N0, N1),
    f3tmp_fft.reshape(N0, N1),
    t, wn, N3.reshape(N0, N1), N4.reshape(N0, N1),
    tmp1.reshape(N0, N1), tmp2.reshape(N0, N1), tmp3.reshape(N0, N1), tmp4.reshape(N0, N1)
)
    k1_w, k1_th = k1_w.ravel(), k1_th.ravel()
    for i in range(N0):
      for j in range(N1):
        jj = i*N1+j
        k1 = wn[i]
        k2 = wn[j]
        k1sq =pow(k1,2)
        k2sq =pow(k2,2)
        ksq = k1sq + k2sq

	      # below is the new u1, u2, theta:
        tmp1 = 1.0 + dt*nu*k1sq
        tmp2 = 1.0 + dt*eta*k2sq
        # print(f" jj: {jj}, nu: {nu}  eta: {eta} k1sq: {k1sq}, k2sq: {k2sq}")

        w[jj] = ( w[jj]  + dt*k1_w[jj] )/tmp1
        th[jj]= ( th[jj] + dt*k1_th[jj])/tmp2
        # print(f"jj, th, k1_th", jj,th[jj], k1_th)
        # input("Press")
    return w.reshape(N0, N1), th.reshape(N0, N1)

def ComputeUVfromVort(w, u1, u2,wn):
    r"""
    work in Fourier space
    -Laplace psi = w, that is, \hat\psi = \hat{w}/|k|^2
    u1 = D_y \psi,  or \hat{u1} = i*k2*\hat{\psi}  = i*k2/|k|^2 \hat{w}
    u2 =-D_x \psi,  or \hat{u1} = -i*k1*\hat{\ps} =-i*k1/|k|^2 \hat{w}
    """

    # If in physical space, move to Fourier for algebra:
    if IN_FOURIER_SPACE[0] == 'n':
        w = np.fft.fftn(w, s=(N0, N1), axes=(0, 1), norm="backward")
    kx = wn[:, None]   # shape (N0, 1)
    ky = wn[None, :]   # shape (1, N1)
    ksq = kx**2 + ky**2
    mask = ksq > 1e-14

    if not np.iscomplexobj(u1):
        u1[:] = u1.astype(np.complex128, copy=False)
    if not np.iscomplexobj(u2):
        u2[:] = u2.astype(np.complex128, copy=False)

    with np.errstate(divide='ignore', invalid='ignore'):
        u1[:] = np.where(mask, (1j * ky / ksq) * w, 0.0)
        u2[:] = np.where(mask, (-1j * kx / ksq) * w, 0.0)

    if IN_FOURIER_SPACE[0] == 'n':
        u1 = np.fft.ifftn(u1,s=(N0, N1), axes=(0, 1), norm="backward").real
        u2 = np.fft.ifftn(u2, s=(N0, N1), axes=(0, 1), norm="backward").real
        w  = np.fft.ifftn(w,  s=(N0, N1), axes=(0, 1), norm="backward").real

    # print(f"[DEBUG] u1 min={u1.real.min():.3e}, max={u1.real.max():.3e}")
    # print(f"[DEBUG] u2 min={u2.real.min():.3e}, max={u2.real.max():.3e}")
    return w, u1, u2

#  k1_w_old, k1_th_old, wo,w,tho,th = do_BDF2(u1, u2,wo, w, wnew, tho, th, thnew,k1_w_old, k1_th_old,k1_w, k1_th, tmp1, tmp2, tmp3, tmp4, g1tmp, f3tmp,N3,N4,wn)


def do_RK2(u1, u2, w, th,k1_w, k1_th, k2_w, k2_th,w_tp, th_tp, u1_tp, u2_tp,g1tmp, f3tmp, t, dt, L, wn, N3, N4, tmp1, tmp2, tmp3, tmp4):
#     # // RK2 method
#     # // has 2 unknowns to solve: w (vorticity) and th (theta)
#     # // work in Fourier space
    x = (np.arange(N0) * L / N0)[:, None]   # (N0, 1)
    y = (np.arange(N1) * L / N1)[None, :]   # (1, N1)
    g1tmp = g1_func(t, x, y, ConvTest, nu).astype(np.complex128)
    f3tmp = f3_func(t, x, y, ConvTest).astype(np.complex128)
    g1tmp = np.fft.fftn(g1tmp, s=(N0, N1), axes=(0, 1), norm="backward")
    f3tmp = np.fft.fftn(f3tmp, s=(N0, N1), axes=(0, 1), norm="backward")
    # k1
    k1_w, k1_th = RHS(k1_w, k1_th, u1,u2, th, w,
                      g1tmp, f3tmp, t,
                      wn, N3,N4, tmp1, tmp2, tmp3, tmp4)
    # tmp state at t+dt using k1
    w_tp  = w + dt * k1_w
    th_tp = th+ dt * k1_th
    # compute u(t+dt) from w(t+dt)
    w_tp, u1_tp, u2_tp = ComputeUVfromVort(w_tp, u1_tp, u2_tp, wn)
    # step2
    g1tmp = g1_func(t + dt, x, y, ConvTest, nu).astype(np.complex128)
    f3tmp = f3_func(t + dt, x, y, ConvTest).astype(np.complex128)
    g1tmp = np.fft.fftn(g1tmp, s=(N0, N1), axes=(0, 1), norm="backward")
    f3tmp = np.fft.fftn(f3tmp, s=(N0, N1), axes=(0, 1), norm="backward")
    # k2
    k2_w, k2_th =RHS(k2_w, k2_th, u1_tp, u2_tp, th_tp, w_tp,
                    g1tmp, f3tmp, t+dt,
                     wn, N3, N4, tmp1, tmp2, tmp3, tmp4)
    # update w, theta
    w  = w  + dt * 0.5 * (k1_w + k2_w)
    th = th + dt * 0.5 * (k1_th + k2_th)
    return w, th

def do_BDF2(u1,u2, wo, w, wnew,tho, th, thnew,k1_w_old, k1_th_old,k1_w, k1_th,tmp1, tmp2, tmp3, tmp4, g1tmp, f3tmp,N3,N4,wn):
    """
    Build RHS in physical space (g1, f3) and them in complex buffers (imag=0)
    Call RHS_BE to fill k1_w, k1_th
    BDF2 update for w and theta in spectral space
    all *_in arrays are 1D complex
    do Backward Differentiation Formula of order 2
    """
    if t>0.9*dt:

      print_w = w.ravel()
      print_wo = wo.ravel()
      print_th = th.ravel()
      print_tho = tho.ravel()
      # k1_w_oldprint,k1_th_oldprint = k1_w_old.ravel(),k1_th_old.ravel()
      for j in range(N1):
        for i in range(N0):
          jj = i*N1+j
          # print("below jj w wo th tho,: " ,jj,print_w[jj].real,print_wo[jj].real ,print_th[jj].real, print_tho[jj].real)
          # input("khvsdjl")


    x = (np.arange(N0) * L / N0)[:, None]
    y = (np.arange(N1) * L / N1)[None, :]

    g1tmp = g1_func(t, x, y, ConvTest, nu) + 0j
    f3tmp = f3_func(t, x, y, ConvTest) + 0j

    g1tmp = np.fft.fftn(g1tmp, s=(N0, N1),axes=(0, 1), norm="backward")
    f3tmp = np.fft.fftn(f3tmp, s=( N0, N1), axes=(0, 1),norm="backward")

    # g1tmpr = g1tmp.ravel()
    # f3tmpr = f3tmp.ravel()
    # # if t>1.9*dt:
    # #     for j in range(N0):
    # #       for i in range(N1):
    # #         jj = i*N1+j
    # #         print(f"jj: {jj} g1_tmpr is {g1tmpr[jj].real} f3tmpr is {f3tmpr[jj].real} at time t: {t}")
    # #         input("enter")

    # if t>1.9*dt:
    #           k1_w_print = k1_w.ravel()
    #           k1_th_print = k1_th.ravel()
    #           print_w = w.ravel()
    #           print_th = th.ravel()
    #           # k1_w_oldprint,k1_th_oldprint = k1_w_old.ravel(),k1_th_old.ravel()
    #           for j in range(N1):
    #             for i in range(N0):
    #               jj = i*N1+j
    #               print("jj, k1_w_old: k1_th_old: ", jj,k1_w_print[jj].real,k1_th_print[jj].real, print_w[jj].real, print_th[jj].real)
    #               input("khvsdjl")


    k1_w, k1_th = RHS_BE(k1_w, k1_th,u1, u2, th, w,g1tmp, f3tmp, t, wn, N3, N4,tmp1, tmp2, tmp3, tmp4)

    k1 = wn[:, None]
    k2 = wn[None, :]
    k1_sq = k1**2
    k2_sq = k2**2

    denom_w  = 1.5 / dt + nu * (k1_sq)
    denom_th = 1.5 / dt + eta * (k2_sq)
    wnew  = (2.0 * k1_w - k1_w_old + (2.0 * w - 0.5 * wo) / dt) / denom_w
    thnew = (2.0 * k1_th - k1_th_old + (2.0 * th - 0.5 * tho) / dt) / denom_th
     # print("dkjfvbjkvsvldsa",t,dt)

    k1_w_old[:] = k1_w
    k1_th_old[:] = k1_th
    wo[:] = w
    tho[:] = th
    w[:] = wnew
    th[:] = thnew

    return k1_w_old, k1_th_old, wo,w,tho,th

def in1_plus_a_in2(out, in1, in2, dt):
    out[:] = in1 + A * in2
    return out

def filter_Krasny(arr: np.ndarray, noise_level: float):
    mask = np.abs(arr) < noise_level
    arr[mask] = 0.0 + 0.0j
    return arr

def filter_exp(in1,Filter_alpha, wn ):
    kx = wn[:, None]   # shape (N0, 1)
    ky = wn[None, :]   # shape (1, N1)

    exp_fx = np.exp(-36.0 * (2.0 * np.abs(kx) / N0) ** Filter_alpha)
    exp_fy = np.exp(-36.0 * (2.0 * np.abs(ky) / N1) ** Filter_alpha)

    exp_filter = exp_fx * exp_fy
    in1[:] *= exp_filter
    return in1

def FindError(u1, u2, th, w):
    """
    Python port of the C FindError function.
    u1, u2, th, w : 1D complex NumPy arrays, length = N0*N1
    Using global N0, N1, L, t, ConvTest and the init functions u1initFunc, u2initFunc, thinitFunc, winitFunc.
    """

    error_u1 = 0.0
    error_u2 = 0.0
    error_th = 0.0
    error_w  = 0.0
    x = (np.arange(N0) * L / N0)[:, None]   # shape (N0, 1)
    y = (np.arange(N1) * L / N1)[None, :]   # shape (1, N1)

    if ConvTest == 1:
        # convergence test
        tmpu1 = abs(u1.real - u1initFunc(t, x, y))
        tmpu2 = abs(u2.real - u2initFunc(t, x, y))
        tmpth = abs(th.real - thinitFunc(t, x, y))
        tmpw  = abs(w.real  - winitFunc(t, x, y))
    else:
        # real application
        tmpu1 = abs(u1.real)
        tmpu2 = abs(u2.real)
        tmpth = abs(th.real)
        tmpw  = abs(w.real)

    error_u1 = np.max(tmpu1)
    error_u2 = np.max(tmpu2)
    error_th = np.max(tmpth)
    error_w  = np.max(tmpw)

    print(f"time           is {t:12.5e}")
    print(f"u1    max norm is {error_u1:12.5e}")
    print(f"u2    max norm is {error_u2:12.5e}")
    print(f"vort  max norm is {error_w:12.5e}")
    print(f"th    max norm is {error_th:12.5e}")


def CompNormsInFourierSpace(u1, u2, th,wn, t, d):
# compute norms in Fourier space
    # reshape for 2D spectral grid
    u1 = u1
    u2 = u2
    th = th

    k1 = wn[:, None]   # shape (N0, 1)
    k2 = wn[None, :]   # shape (1, N1)
    k1sq = k1**2
    k2sq = k2**2
    ksq  = k1sq + k2sq

    # |u|^2 and |th|^2
    absu2 = np.abs(u1)**2 + np.abs(u2)**2
    absth2 = np.abs(th)**2
    # DFT of tilde{u}(x,y)= u(x,y) - \bar{u}(y) compute H1 norm
    mask = np.abs(k1) > 1e-14
    tilde_u = absu2 * mask
    tilde_th = absth2 * mask

    # L2 norms
    u_L2 = absu2.sum()
    th_L2 = absth2.sum()
    tilde_u_L2 = tilde_u.sum()
    tilde_th_L2 = tilde_th.sum()

    # H1 / H2 norms
    factor = 1 + k1sq + k2sq
    u_H2sq  = (absu2 * (factor + k1sq**2 + k2sq**2 + k1sq*k2sq)).sum()
    th_H1sq = (absth2 * factor).sum()
    th_H2sq = (absth2 * (factor + k1sq**2 + k2sq**2 + k1sq*k2sq)).sum()

    tilde_u_H1sq = (tilde_u * factor).sum()
    tilde_th_H1sq = (tilde_th * factor).sum()
    D2_tilde_th_H2sq = (tilde_th * (factor + k1sq**2 + k2sq**2 + k1sq*k2sq) * k2sq).sum()

    # Divergence
    DIV = np.abs(k1*u1 + k2*u2)**2
    DIV = np.sqrt(DIV.sum())

    # two scalars used in Parseval-Plancherel identity in integrals:
    tmp4 = 2*np.pi / (N1*N0)
    tmp5 = tmp4**2

    # final quantities
    WuQ2 = np.sqrt(tilde_u_H1sq*tmp5) + np.sqrt(tilde_th_H1sq*tmp5)
    WuQ3 = t * (tilde_u_H1sq*tmp5 + tilde_th_H1sq*tmp5)

    ThQ1 = tmp4 * np.sqrt(th_L2)
    ThQ2 = tmp4 * np.sqrt(tilde_th_L2)
    ThQ3 = np.sqrt(tilde_th_H1sq*tmp5)

    UQ1  = tmp4 * np.sqrt(u_L2)
    UQ2  = tmp4 * np.sqrt(tilde_u_L2)
    UQ3  = np.sqrt(tilde_u_H1sq*tmp5)

    # Integrals
    Integral1 = 2*nu*dt*(k1sq*u_H2sq).sum() if t >= 1e-10 else 0.0
    Integral2 = 2*eta*dt*D2_tilde_th_H2sq*tmp5 if t >= 1e-10 else 0.0
    Integral3 = dt*(k1sq*th_L2).sum() if t >= 1e-10 else 0.0

    WuQ1 = (u_H2sq + th_H2sq)*tmp5 + Integral1 + Integral2 + Integral3

    return WuQ1, WuQ2, WuQ3, ThQ1, ThQ2, ThQ3, UQ1, UQ2, UQ3, Integral1, Integral2, Integral3, DIV*tmp4, np.sqrt(th_H1sq)*tmp4, np.sqrt(th_H2sq*tmp5)

def read_input(filename):
    config = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    config.read(filename)

    p = config["parameters"]

    cfg = {}
    cfg["METHOD"] = int(p["METHOD"])
    cfg["TMAX"] = float(p["TMAX"])
    cfg["dt"] = float(p["dt"])
    cfg["dt_print"] = float(p["dt_print"])
    cfg["dt_norms"] = float(p["dt_norms"])
    cfg["eta"] = float(p["eta"])
    cfg["nu"] = float(p["nu"])
    cfg["WuEpsi"] = float(p["WuEpsi"])
    cfg["N0"] = int(p["N0"])
    cfg["N1"] = int(p["N1"])
    cfg["restart"] = p["restart"]
    cfg["irestart"] = int(p["irestart"])
    cfg["ConvTest"] = int(p["ConvTest"])
    cfg["ShenYang"] = int(p["ShenYang"])
    cfg["USE_Filter"] = int(p["USE_Filter"])
    cfg["Filter_alpha"] = float(p["Filter_alpha"])
    cfg["Filter_noiselevel"] = float(p["Filter_noiselevel"])

    return cfg

def printreal(u1, u2, th, w,
              t, Integral1, Integral2, Integral3,
              iout, CFL_break, N0, N1):
    """
    Python equivalent of the C printreal function.
    u1, u2, th, w : 1D NumPy arrays of dtype=complex128, length = N0*N1
    t, Integral1, Integral2, Integral3 : scalars
    iout : output index (int)
    CFL_break : flag (0 or 1)
    N0, N1 : grid dimensions
    """
    u1 = u1.ravel()
    u2 = u2.ravel()
    th = th.ravel()
    w  = w.ravel()
    # choose filename
    if CFL_break == 0:
        if iout < 10:
            fn = f"data000{iout}"
        elif iout < 100:
            fn = f"data00{iout:02d}"
        elif iout < 1000:
            fn = f"data0{iout:03d}"
        else:
            fn = f"data{iout:04d}"
    else:
        fn = "dataCFLb"

    print(f" file name = {fn}")

    with open(fn, "w") as f:
        # write header
        f.write(f"{t:.20f}  {Integral1:.20f}  {Integral2:.20f}  {Integral3:.20f}\n")

        # write data
        for j in range(N1):
            for i in range(N0):
                jj = i * N1 + j
                # real parts
                f.write(f"{u1[jj].real:.20f}  {u2[jj].real:.20f}  "
                        f"{th[jj].real:.20f}  {w[jj].real:.20f}\n")
                # imaginary parts
                f.write(f"{u1[jj].imag:.20f}  {u2[jj].imag:.20f}  "
                        f"{th[jj].imag:.20f}  {w[jj].imag:.20f}\n")
            f.write("\n")

def read_data(u1, u2, th, w,irestart):
    # construct filename
    if irestart < 10:
        fn = f"data000{irestart}"
    elif irestart < 100:
        fn = f"data00{irestart:02d}"
    elif irestart < 1000:
        fn = f"data0{irestart:03d}"
    else:
        fn = f"data{irestart:04d}"

    print(f"reading {fn}")

    with open(fn, "r") as f:
        # read header line
        header = f.readline().strip().split()
        t, Integral1, Integral2, Integral3 = map(float, header)
        # read array values
        for j in range(N1):
            for i in range(N0):
                jj = i * N1 + j
                # read real parts
                vals_real = list(map(float, f.readline().split()))
                # read imaginary parts
                vals_imag = list(map(float, f.readline().split()))

                u1[jj] = vals_real[0] + 1j*vals_imag[0]
                u2[jj] = vals_real[1] + 1j*vals_imag[1]
                th[jj] = vals_real[2] + 1j*vals_imag[2]
                w[jj]  = vals_real[3] + 1j*vals_imag[3]

            f.readline()  # skip the blank line

    return t, Integral1, Integral2, Integral3

def DefineWnWnabs(N0, N1):
    wn = np.zeros(N0, dtype=float)
    for i in range(N0):
        if i <= N0 // 2:
            wn[i] = float(i)
        else:
            wn[i] = float(i - N0)
    wn_abs_local = np.abs(wn)
    return wn, wn_abs_local

def ComputeThetaAverage(th):
      if IN_FOURIER_SPACE[0] == 'y':
        th = np.fft.ifftn(th, s=(N0, N1), axes=(0, 1), norm="backward").real
      InitAverage = 0.0
      #compute InitAverage:
      if t < 1e-10:
        InitAverage = th.mean(axis=0)

      tmp = th.real.mean(axis=0)
      ff = tmp - InitAverage
      AveErrmaxtoinit = tmp.max()

      if IN_FOURIER_SPACE[0] == 'y':
        th = np.fft.fftn(th,s=(N0, N1), axes=(0, 1), norm="backward")

      return AveErrmaxtoinit

def SetExactVort(w, time_spec):

      if( IN_FOURIER_SPACE[0]=='y'):
        #   convert to physical space:
        w =np.fft.ifftn(w, s=(N0, N1), axes=(0,1), norm="backward").real

      x = (np.arange(N0) * L / N0)[:, None]
      y = (np.arange(N1) * L / N1)[None, :]

      w = winitFunc(time_spec, x, y).astype(np.complex128)
      w.imag[:] = 0.0
      if IN_FOURIER_SPACE[0] == 'y':
          w = np.fft.fftn(w, norm="backward")
      return w

def output_data(u1, u2, th, w):
      print("first test IOUT",iout)
      if( IN_FOURIER_SPACE[0]=='y'):
        # Convert to physical space (inverse FFT + normalization)
        u1 = np.fft.ifftn(u1,s=(N0, N1), axes=(0, 1), norm="backward").real
        u2 = np.fft.ifftn(u2,s=(N0, N1),   axes=(0, 1),norm="backward").real
        th = np.fft.ifftn(th,s=(N0, N1), axes=(0, 1), norm="backward").real
        w  = np.fft.ifftn(w, s=(N0, N1),  axes=(0, 1),norm="backward").real

      printreal(u1, u2, th, w,t, Integral1, Integral2, Integral3, iout, CFL_break, N0, N1);
      FindError(u1, u2, th, w)

      if( IN_FOURIER_SPACE[0]=='y'):
        #   convert to Fourier space:
        u1 = np.fft.fftn(u1,s=(N0, N1), axes=(0, 1), norm="backward")
        u2 = np.fft.fftn(u2,s=(N0, N1), axes=(0, 1), norm="backward")
        th = np.fft.fftn(th,s=(N0, N1), axes=(0, 1), norm="backward")
        w  = np.fft.fftn(w, s=(N0, N1), axes=(0, 1), norm="backward")
      # return u1, u2, th, w

def InitVariables(w, th):

    x = (np.arange(N0) * L / N0)[:, None]
    y = (np.arange(N1) * L / N1)[None, :]

    w[:, :] = winitFunc(0.0, x, y) + 0j
    th[:, :] = thinitFunc(0.0, x, y) + 0j
    return w, th

def CompCFLcondition(u1, u2,IN_FOURIER_SPACE, dt, L):
    """
    Python equivalent of the C CompCFLcondition.
    Keeps same names and structure.
    """
    if IN_FOURIER_SPACE[0] == 'y':
        u1_phys = np.fft.ifftn(u1, s=(N0, N1), axes=(0, 1), norm="backward").real
        u2_phys = np.fft.ifftn(u2, s=(N0, N1), axes=(0, 1), norm="backward").real

    #  umax on root
    umax = np.max(np.sqrt(np.abs(u1_phys)**2 + np.abs(u2_phys)**2))
    # CFL condition
    h = L / N0
    tmp = dt * umax / h
    CFL_break = 0
    if tmp > 0.5:
        CFL_break = 1
        print("***********************************")
        print(f"    dt        = {dt:12.5e}")
        print(f"    umax      = {umax:12.5e}")
        print(f"    h         = {h:12.5e}")
        print(f"CFL=dt*umax/h = {tmp:12.5e}")
        print("CFL is too big, need to decrease dt")
        print("***********************************")

    return CFL_break, umax, h

def test_909(w):

    w[0] += 200
    w[1] += 400
    return w
def MakeAverageZero(*input):

    out_arrays = []
    for arr in input:

    # this average is the one in the whole domain
        if IN_FOURIER_SPACE[0] == 'y':
            # Convert to physical space:
            arr = np.fft.ifftn(arr, s=(N0, N1), axes=(0, 1), norm="backward").real

        # Compute and subtract average:
        ave = arr.mean()
        print(f"average = {ave:12.5e}")
        arr = arr - ave
        # Convert back to Fourier space
        if IN_FOURIER_SPACE[0] == 'y':
            arr = np.fft.fftn(arr,s=(N0, N1), axes=(0, 1), norm="backward")

        out_arrays.append(arr)

    return tuple(out_arrays)

def InitFFTW(wo, w, tho, th,
             u1o, u1, u2o, u2):
    wo  = np.fft.fftn(wo, s=(N0, N1), axes=(0, 1), norm="backward")
    w  = np.fft.fftn(w, s=(N0, N1), axes=(0, 1), norm="backward")
    tho= np.fft.fftn(tho,s=(N0, N1), axes=(0, 1), norm="backward")
    th = np.fft.fftn(th, s=(N0, N1), axes=(0, 1), norm="backward")
    u1o= np.fft.fftn(u1o,s=(N0, N1), axes=(0, 1), norm="backward")
    u1 = np.fft.fftn(u1, s=(N0, N1), axes=(0, 1), norm="backward")
    u2o = np.fft.fftn(u2o, s=(N0, N1), axes=(0, 1), norm="backward")
    u2  = np.fft.fftn(u2, s=(N0, N1), axes=(0, 1), norm="backward")

    return wo, w, tho, th, u1o, u1, u2o, u2

def _split_rows(total_rows: int, size: int, rank: int):
  """Even block row decomposition like FFTW MPI would do."""
  base = total_rows // size
  extra = total_rows % size
  n0 = base + (1 if rank < extra else 0)
  start = rank * base + min(rank, extra)
  return n0, start

N0 = N1 = None
METHOD = None
TMAX = None
dt = None
dt_print = None
dt_norms = None
eta = None
nu = None
WuEpsi = None
restart = None
irestart = None
ConvTest = None
ShenYang = None
USE_Filter = None
Filter_alpha = None
Filter_noiselevel = None

nprocs = None
IN_FOURIER_SPACE = ['n']
wn_abs_local = None

t = 0.0

iout = 0
iter_print = 0
iter_norms = 0
CFL_break = 0

Integral1 = Integral2 = Integral3 = 0.0
WuQ1 = WuQ2 = WuQ3 = 0.0
ThQ1 = ThQ2 = ThQ3 = 0.0
UQ1 = UQ2 = UQ3 = 0.0
DIV = 0.0
umax = 0.0
AveErrmaxtoinit = 0.0

def main():
    global N0, N1, METHOD, TMAX, dt, dt_print, dt_norms, eta, nu, WuEpsi
    global restart, irestart, ConvTest, ShenYang, USE_Filter, Filter_alpha
    global Filter_noiselevel, IN_FOURIER_SPACE, wn_abs, iout, t
    global iter_print, iter_norms, CFL_break, Integral1, Integral2, Integral3
    global WuQ1, WuQ2, WuQ3, ThQ1, ThQ2, ThQ3, UQ1, UQ2, UQ3, DIV, umax
    global AveErrmaxtoinit, w, th, wn

    cfg = read_input("/content/drive/MyDrive/input.ini")

    # install globals from config
    METHOD           = cfg['METHOD']
    TMAX             = cfg['TMAX']
    dt               = cfg['dt']
    dt_print         = cfg['dt_print']
    dt_norms         = cfg['dt_norms']
    eta              = cfg['eta']
    nu               = cfg['nu']
    WuEpsi           = cfg['WuEpsi']
    N0               = cfg['N0']
    N1               = cfg['N1']
    restart          = cfg['restart']
    irestart         = cfg['irestart']
    ConvTest         = cfg['ConvTest']
    ShenYang         = cfg['ShenYang']
    USE_Filter       = cfg['USE_Filter']
    Filter_alpha     = cfg['Filter_alpha']
    Filter_noiselevel= cfg['Filter_noiselevel']

    max_iter = int(math.ceil(TMAX / dt))
    iter_print = int(math.ceil(dt_print / dt))
    iter_norms = int(math.ceil(dt_norms / dt))
    u1   = np.zeros((N0, N1), dtype=np.complex128)
    u1o  = np.zeros((N0, N1), dtype=np.complex128)
    u2   = np.zeros((N0, N1), dtype=np.complex128)
    u2o  = np.zeros((N0, N1), dtype=np.complex128)
    th   = np.zeros((N0, N1), dtype=np.complex128)
    tho  = np.zeros((N0, N1), dtype=np.complex128)
    thn  = np.zeros((N0, N1), dtype=np.complex128)
    thnew   = np.zeros((N0, N1), dtype=np.complex128)
    w    = np.zeros((N0, N1), dtype=np.complex128)
    wo   = np.zeros((N0, N1), dtype=np.complex128)
    wnew   = np.zeros((N0, N1), dtype=np.complex128)

    k1_w_old = np.zeros((N0, N1), dtype=np.complex128)
    k1_th_old = np.zeros((N0, N1), dtype=np.complex128)
    k1_w = np.zeros((N0, N1), dtype=np.complex128)
    k1_th = np.zeros((N0, N1), dtype=np.complex128)
    k2_w = np.zeros((N0, N1), dtype=np.complex128)
    k2_th = np.zeros((N0, N1), dtype=np.complex128)

    w_tp = np.zeros((N0, N1), dtype=np.complex128)
    th_tp = np.zeros((N0, N1), dtype=np.complex128)
    u1_tp = np.zeros((N0, N1), dtype=np.complex128)
    u2_tp = np.zeros((N0, N1), dtype=np.complex128)

    g1tmp = np.zeros((N0, N1), dtype=np.complex128)
    f3tmp = np.zeros((N0, N1), dtype=np.complex128)
    N3 = np.zeros((N0, N1), dtype=np.complex128)
    N4 = np.zeros((N0, N1), dtype=np.complex128)
    tmp1 = np.zeros((N0, N1), dtype=np.complex128)
    tmp2 = np.zeros((N0, N1), dtype=np.complex128)
    tmp3 = np.zeros((N0, N1), dtype=np.complex128)
    tmp4 = np.zeros((N0, N1), dtype=np.complex128)

    print(f"TMAX, MAX_ITER = {TMAX:.6f}, {max_iter:.6f}")
    print(f"dt_print = {dt_print:.6f}, iter_print={iter_print}")
    print(f"dt_norms = {dt_norms:.6f}, iter_norms={iter_norms}")
    print(f"dt = {dt:.6e}")
    print(f"N0, N1 = {N0}, {N1}")
    print(f"hx, hy = {L/N0:12.5e}, {L/N1:12.5e}")
    print(f"N0    = {N0}")

    # if these will hold Fourier-space values at any point, make them complex
    wn, wn_abs_local = DefineWnWnabs(N0, N1)
    t = 0.0
    iout = 0
    iter_start = 0
    if restart[0].lower() == 'n':
        # root prepares initial w, theta
        if ConvTest == 1:
          w,th = InitVariables(w, th)
        else:
          w = Read_VorTemp()

        th, u1, u2 = MakeAverageZero(th, u1, u2)

        IN_FOURIER_SPACE[0] = 'n'
        w, u1, u2 = ComputeUVfromVort(w, u1, u2, wn)

        output_data(u1, u2, th, w)

    elif restart[0].lower() == 'y':
            # read into global arrays on root
        t, Integral1, Integral2, Integral3 = read_data(u1, u2, th, w, irestart)
        iout        = int(math.ceil(t / dt_print))
        iter_start  = int(math.ceil(t / dt))

        print("restart info *********************")
        print(f"restart      = {restart}")
        print(f"restart time = {t:.6e}")
        print(f"iout         = {iout}")
        print(f"iter_start   = {iter_start}")

        IN_FOURIER_SPACE[0] = 'n'
        w, u1, u2 = ComputeUVfromVort(w, u1, u2, wn)
    else:
        print("This input for restart is invalid")
        return

    # print_wo = wo.ravel()
    # print_w = w.ravel()
    # print_th = th.ravel()
    # print_tho = tho.ravel()
    # k1_w_oldprint,k1_th_oldprint = k1_w_old.ravel(),k1_th_old.ravel()
    # for j in range(N1):
    #   for i in range(N0):
    #     jj = i*N1+j
    #     print("jj, k1_w_old: k1_th_old: ", jj,k1_w_oldprint[jj].real,k1_th_oldprint[jj].real, print_wo[jj].real, print_w[jj].real)
    #     input("khvsdjl")


    #  go to Fourier space
    wo, w, tho, th, u1o, u1, u2o, u2 = InitFFTW(wo, w, tho, th, u1o, u1, u2o, u2)

    IN_FOURIER_SPACE[0] = 'y'

    #  test initial norms
    (WuQ1, WuQ2, WuQ3,ThQ1, ThQ2, ThQ3, UQ1, UQ2, UQ3, Integral1, Integral2, Integral3,DIV, th_H1, th_H2) = CompNormsInFourierSpace(u1, u2, th,wn, t, dt)

    with open("norms", "a") as f:
        f.write(f"{t:12.5e}   {WuQ1:12.5e}   {WuQ2:12.5e}   {WuQ3:12.5e}   "
                f"{DIV:12.5e}  {umax:12.5e}  {AveErrmaxtoinit:12.5e}\n")
    with open("morenorms", "a") as f:
        f.write(f"{t:12.5e}   {ThQ1:12.5e}   {ThQ2:12.5e}   {ThQ3:12.5e}   "
                f"{UQ1:12.5e}  {UQ2:12.5e}  {UQ3:12.5e}\n")
        print(f"ThQ1={ThQ1:12.5e},  ThQ2={ThQ2:12.5e}")

    #  main time loop
    for it in range(iter_start, max_iter):

        CFL_break, umax, h = CompCFLcondition(
            u1, u2,
            IN_FOURIER_SPACE, dt, L
        )

        if CFL_break == 1:
           output_data(u1, u2, th, w )

           print("CFL violated; aborting.")
           return

        # time stepper
        if METHOD == 1:
            w, th = do_IMEX(k1_w, k1_th, u1, u2, th, w, g1tmp, f3tmp, t, wn, N3,N4,tmp1, tmp2, tmp3, tmp4)
            w, u1, u2 = ComputeUVfromVort(w, u1, u2, wn)

        elif METHOD == 2:  # BDF2

            if it == iter_start:

              wo,tho, u1o, u2o = w, th, u1, u2
              # print_wo = wo.ravel()
              # print_w = w.ravel()
              # print_th = th.ravel()
              # print_tho = tho.ravel()
              # k1_w_oldprint,k1_th_oldprint = k1_w_old.ravel(),k1_th_old.ravel()
              # for j in range(N1):
              #   for i in range(N0):
              #     jj = i*N1+j
              #     print("jj, k1_w_old: k1_th_old: ", jj,k1_w_oldprint[jj].real,k1_th_oldprint[jj].real, print_wo[jj].real, print_w[jj].real, t)
              #     input("khvsdjl")
              k1_w_old, k1_th_old = RHS_k1_w_th(u1o, u2o, tho, wo, k1_w_old, k1_th_old, dt,t, g1tmp, f3tmp, N3,N4, tmp1, tmp2, tmp3, tmp4)

              w, th = do_RK2(u1, u2, w, th,
                           k1_w, k1_th, k2_w, k2_th,
                           w_tp, th_tp, u1_tp, u2_tp,
                           g1tmp, f3tmp,
                           t, dt, L,
                           wn, N3,N4, tmp1, tmp2, tmp3, tmp4)

              w, u1, u2 = ComputeUVfromVort(w, u1, u2, wn)

            else:

              k1_w_old, k1_th_old, wo,w,tho,th = do_BDF2(u1, u2,wo, w, wnew, tho, th, thnew,k1_w_old, k1_th_old,k1_w, k1_th, tmp1, tmp2, tmp3, tmp4, g1tmp, f3tmp,N3,N4,wn)
              w, u1, u2 = ComputeUVfromVort(w, u1, u2,wn)

        elif METHOD == 3:  # RK2

            w, th = do_RK2(u1, u2, w, th,
                           k1_w, k1_th, k2_w, k2_th,
                           w_tp, th_tp, u1_tp, u2_tp,
                           g1tmp, f3tmp,
                           t, dt, L,
                           wn, N3,N4, tmp1, tmp2, tmp3, tmp4)
            w, u1, u2 = ComputeUVfromVort(w, u1, u2, wn)

        # advance time
        t += dt
        if USE_Filter == 1:
            u1 = filter_exp(u1, Filter_alpha,wn)
            u2 = filter_exp(u2, Filter_alpha,wn)
            th = filter_exp(th, Filter_alpha,wn)
            w = filter_exp(w,  Filter_alpha,wn)
        elif USE_Filter == 2:
            u1 = filter_Krasny(u1, Filter_noiselevel)
            u2 = filter_Krasny(u2, Filter_noiselevel)
            th = filter_Krasny(th, Filter_noiselevel)
            w = filter_Krasny(w,  Filter_noiselevel)
        # norms
        (WuQ1, WuQ2, WuQ3,ThQ1, ThQ2, ThQ3, UQ1, UQ2, UQ3, Integral1, Integral2, Integral3,DIV, th_H1, th_H2) = CompNormsInFourierSpace(u1, u2, th,wn, t, dt)

        AveErrmaxtoinit= ComputeThetaAverage(th)
        if (it + 1) % iter_norms == 0:
            with open("norms", "a") as f:
                f.write(f"{t:12.5e}   {WuQ1:12.5e}   {WuQ2:12.5e}   {WuQ3:12.5e}   "
                        f"{DIV:12.5e}  {umax:12.5e}  {AveErrmaxtoinit:12.5e}\n")
            with open("morenorms", "a") as f:
                f.write(f"{t:12.5e}   {ThQ1:12.5e}   {ThQ2:12.5e}   {ThQ3:12.5e}   "
                        f"{UQ1:12.5e}  {UQ2:12.5e}  {UQ3:12.5e}\n")
                print(f"ThQ1={ThQ1:12.5e},  ThQ2={ThQ2:12.5e}")

        # periodic output
        if (it + 1) % iter_print == 0:
            iout += 1

            print(f"iter, time = {it+1}, {t}")
            output_data(u1, u2, th, w)


if __name__ == "__main__":
    main()
