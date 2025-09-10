"""
August 13, 2025
 * Xiaoming Zheng
 * Edited by Akram Moustafa
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
   N5 = - (D_t - eta D_22) P(u\cdot \nabla u)     + \nabla^\perp D_1 \Laplace^{-1} (u\cdot \nabla theta)
   N6 = - (D_t - nu  D_11) (u\cdot \nabla theta)  + u\cdot \nabla u_2 - D_2 Laplace^{-1} \nabla (u\cdot \nabla u)

where P is Leray's projection, that is,
   P(u) = u - \nabla Laplace^{-1} (\nabla\cdot u)

Method: convert to first order differential system on t: introducing new variables
   v1  = D_t u1
   v2  = D_t u2
   ps = D_t theta
"""

import numpy as np
def g1_func(t, x, y,ConvTest, nu):

    if(ConvTest):
        u1, u2, D1w, D2w, D11w, D1th, Dtw;
        # //w=  -cos(t)*2*a*cos(2*a*x)*pow(sin(a*y),2)  -cos(t)*pow(sin(a*x),2)*2*a*cos(2*a*y)
        # //theta= cos(t)*pow(sin(a*x),2)*sin(2*a*y)

        u1   = math.cos(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)
        u2   =-math.cos(t)*math.sin(2*A*x)*pow(math.sin(A*y),2)
        Dtw  = math.cos(t)*2*A*math.cos(2*A*x)*pow(math.sin(A*y),2) +math.sin(t)*pow(math.sin(a*x),2)*2*a*math.cos(2*a*y)
        D1w  = math.cos(t)*4*A*A*math.sin(2*A*x)*pow(math.sin(A*y),2)-math.cos(t)*math.sin(2*a*x)*2*a*a*math.cos(2*a*y)
        D2w  =-math.cos(t)*2*A*A*math.cos(2*A*x)*math.sin(2*A*y) + math.cos(t)*pow(math.sin(a*x),2)*4*a*a*math.sin(2*a*y)
        D11w = math.cos(t)*8*A*A*A*math.cos(2*A*x)*pow(math.sin(A*y),2) -math.cos(t)*math.cos(2*a*x)*4*a*a*a*math.sin(2*a*y)
        D1th = math.cos(t)*A*math.sin(2*A*x)*math.sin(2*A*y)

        return Dtw + u1*D1w + u2*D2w - D1th - nu*D11w
    else:
        return 0

def f3_func( t,  x,  y, ConvTest):

    if( ConvTest):

        u1   = math.cos(t)*pow(math.sin(a*x),2)*math.sin(2*a*y)
        u2   =-math.cos(t)*math.sin(2*a*x)*pow(math.sin(a*y),2)
        th   = u1
        Dtth =-math.sin(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)
        D1th = math.cos(t)*A*math.sin(2*A*x)*math.sin(2*A*y)
        D2th = math.cos(t)*pow(math.sin(A*x),2)*2*A*math.cos(2*A*y)
        D22th=-math.cos(t)*pow(math.sin(A*x),2)*4*A*A*math.sin(2*A*y)

        return Dtth + u1*D1th + u2*D2th + u2- eta*D22th
    else:
        return 0
def f2_func( t,  x,  y, a, ConvTest):
    if( ConvTest ):

        u1   = math.cos(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)
        u2   =-math.cos(t)*math.sin(2*A*x)*pow(math.sin(A*y),2)
        th   = u1
        p   = math.cos(t)*math.sin(2*A*x)*math.sin(2*A*y)
        Dtu2 = math.sin(t)*math.sin(2*A*x)*pow(math.sin(A*y),2)
        D1u2 =-math.cos(t)*2*A*math.cos(2*A*x)*pow(math.sin(A*y),2)
        D2u2 =-math.cos(t)*math.sin(2*A*x)*A*math.sin(2*A*y)
        D11u2= math.cos(t)*4*A*A*math.sin(2*A*x)*pow(math.sin(A*y),2)
        D2p  = math.cos(t)*math.sin(2*A*x)*2*A*math.cos(2*A*y)

        return Dtu2 +u1*D1u2 + u2*D2u2 + D2p - nu*D11u2 - th
    else:
        return 0

def f1_func( t,  x,  y, ConvTest):

    if( ConvTest ):

        u1   = math.cos(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)
        u2   =-math.cos(t)*math.sin(2*A*x)*pow(math.sin(A*y),2)
        p   =  math.cos(t)*math.sin(2*A*x)*math.sin(2*A*y)
        Dtu1 =-math.sin(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)
        D1u1 = math.cos(t)*A*math.sin(2*A*x)*math.sin(2*A*y)
        D2u1 = math.cos(t)*pow(math.sin(A*x),2)*2*A*math.cos(2*A*y)
        D11u1= math.cos(t)*2*A*A*math.cos(2*A*x)*math.sin(2*A*y)
        D1p  = math.cos(t)*2*A*math.cos(2*A*x)*math.sin(2*A*y)

        return Dtu1 +u1*D1u1 + u2*D2u1 + D1p - nu*D11u1
    else:
        return 0
def winitFunc( t,  x,  y):

  if(ConvTest):
     return -math.cos(t)*2*A*math.cos(2*A*x)*pow(math.sin(A*y),2)-math.cos(t)*pow(math.sin(A*x),2)*2*A*math.cos(2*A*y)
  else:
     return -WuEpsi*math.cos(t)*2*A*math.cos(2*A*x)*pow(math.sin(A*y),2)-WuEpsi*math.cos(t)*pow(math.sin(A*x),2)*2*A*math.cos(2*A*y);


def prinitFunc( t,  x,  y):
  if(ConvTest):
     return math.cos(t)*math.sin(2*A*x)*math.sin(2*A*y);
  else:
     return WuEpsi*math.cos(t)*pow(math.sin(A*x),2)*math.sin(2*A*y);

def thinitFunc( t,  x,  y):
# // initial value of theta
  if(ConvTest):
     return math.cos(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)
  else:
     nX=4
     tmp0 = 4*(1.e0 + 1.e0/4 + 1.e0/9 + 1.e0/16)
     tmp1 = tmp0
     tmp2 = tmp0
     for i in range(1,nX+1):
         tmp1 -= 4.e0/(i*i)*math.cos(i*x)
         tmp2 -= 4.e0/(i*i)*math.cos(i*y)
    #  tmp1 is the partial sum of 2pi^2/3- x(2pi-x) on [0,2pi]
     return WuEpsi*math.cos(t)* (tmp1 * tmp2 - tmp0*tmp0)


def u1initFunc( t,  x,  y):
  if(ConvTest):
     return math.cos(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)
  else:
     return WuEpsi*math.cos(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)


def u2initFunc( t,  x,  y):
  if(ConvTest):
     return -math.cos(t)*math.sin(2*A*x)*pow(math.sin(A*y),2)
  else:
     return -WuEpsi*math.cos(t)*math.sin(2*A*x)*pow(math.sin(A*y),2)

def AfterNonlinear( in_u1, in_u2, in_w, in_th, time_spec):

    in_u1 = np.fft.fftn(in_u1, norm=None).ravel()
    in_u2 = np.fft.fftn(in_u2, norm=None).ravel()
    in_th = np.fft.fftn(in_th, norm=None).ravel()
    in_w = np.fft.fftn(in_w, norm=None).ravel()

    return in_u1, in_u2, in_w, in_th

def PreNonlinear( in_u1, in_u2, in_w, in_th,
                  N4, N3,  time_spec):

    # Above all, convert input quantities to Physical space:
    in_u1 = np.fft.ifftn(in_u1.reshape(local_n0, N1), norm=None).real.ravel()
    in_u2 = np.fft.ifftn(in_u2.reshape(local_n0, N1), norm=None).real.ravel()
    in_th = np.fft.ifftn(in_th.reshape(local_n0, N1), norm=None).real.ravel()
    in_w  = np.fft.ifftn(in_w.reshape(local_n0, N1), norm=None).real.ravel()
    # divide them by DIM because FFTW does not perform it:
    fftw_normalize(in_u1, DIM)
    fftw_normalize(in_u2, DIM)
    fftw_normalize(in_th, DIM)
    fftw_normalize(in_w, DIM)
    # initialize N5 and N6:
    for i in range(local_n0):
        for j in range(N1):
            jj = i * N1 + j
            N4[jj] = 0.0 + 0.0j
            N3[jj] = 0.0 + 0.0j

    return N3, N4,in_u1, in_u2, in_w, in_th
def Nonlinear3and4(in_u1, in_u2, in_w, in_th, tmp1,  tmp2,  tmp3, tmp4, N4,  N3, time_spec):

    # computer N3=- (u\cdot\nabla)\theta
    # computer N4=- (u\cdot\nabla) w
    # Denote z = hat (u\cdot nabla)\theta = hat( div(u*theta) ):
    #          = hat( D_1(u1*theta) + D_2(u2*theta) )

    # first we need u1*theta, u1*theta in Physical space
                for i in range(local_n0):
                    for j in range(N1):
                        jj = i*N1+j;
            #    set u1 and u2 as exact values to check on p:
            #    set u1 and u2 as exact values to check on p:
            #    set u1 and u2 as exact values to check on p:
            #    set u1 and u2 as exact values to check on p:

                jj=i*N1+j
                x=L*(local_start+i)/N0
                y=L*j/N1

                tmp1[jj] = in_u1[jj].real * in_th[jj].real + 0.0j  # u1*theta, real only
                tmp2[jj] = in_u2[jj].real * in_th[jj].real + 0.0j  # u2*theta
                tmp3[jj] = in_u1[jj].real * in_w[jj].real   + 0.0j # u1*vorticity
                tmp4[jj] = in_u2[jj].real * in_w[jj].real   + 0.0j # u2*vorticity

    #    second we compute the DFT of u1*theta, u2*theta:
                tmp1 = np.fft.fftn(tmp1.reshape(local_n0, N1), norm=None).ravel()
                tmp2 = np.fft.fftn(tmp2.reshape(local_n0, N1), norm=None).ravel()
                tmp3 = np.fft.fftn(tmp3.reshape(local_n0, N1), norm=None).ravel()
                tmp4 = np.fft.fftn(tmp4.reshape(local_n0, N1), norm=None).ravel()
    #    third, compute N3 and N4:
                for i in range(local_n0):
                    for j in range(N1):
                        jj = i*N1+j;                    # position in the 1-D local array
                        k1 = wn[local_start + i]
                        k2 = wn[j]
                        ksq = pow(k1,2) + pow(k2,2)

                        # N3 = z = hat( D_1(u1*theta) + D_2(u2*theta) )
                        z = (-k1*tmp1[jj].imag - k2*tmp2[jj].imag)  +  (k1*tmp1[jj].real + k2*tmp2[jj].real)
                        N3[jj] += z


                        # N4 = z = hat( D_1(u1*vort) + D_2(u2*vort) )
                        z = (-k1*tmp3[jj].imag - k2*tmp4[jj].imag) +  (k1*tmp3[jj].real + k2*tmp4[jj].real)
                        N4[jj]= z

                return tmp1, tmp2, tmp3, tmp4, N4, N3

def do_RK2(u1_in, u2_in, w_in, th_in):
#     # // RK2 method
#     # // has 2 unknowns to solve: w (vorticity) and th (theta)
#     # // work in Fourier space
    size = local_n0 * N1

    # 1-D complex arrays
    k1_w  = np.empty(size, dtype=np.complex128)
    k1_th = np.empty(size, dtype=np.complex128)
    k2_w  = np.empty(size, dtype=np.complex128)
    k2_th = np.empty(size, dtype=np.complex128)

    w_tp  = np.empty(size, dtype=np.complex128)
    th_tp = np.empty(size, dtype=np.complex128)
    u1_tp = np.empty(size, dtype=np.complex128)
    u2_tp = np.empty(size, dtype=np.complex128)

    g1tmp_local = np.empty(size, dtype=np.complex128)
    f3tmp_local = np.empty(size, dtype=np.complex128)

    # step 1: build forcing in physical space
    for i in range(local_n0):
        x = L*(local_start + i)/N0
        for j in range(N1):
            jj = i*N1 + j
            y = L*j/N1
            g1tmp_local[jj] = g1_func(t, x, y, ConvTest, nu) + 0.0j
            f3tmp_local[jj] = f3_func(t, x, y, ConvTest)     + 0.0j

    # FFT -> spectral
    g1tmp_local = np.fft.fftn(g1tmp_local.reshape(local_n0, N1), norm=None).ravel()
    f3tmp_local = np.fft.fftn(f3tmp_local.reshape(local_n0, N1), norm=None).ravel()

    # k1
    RHS(k1_w, k1_th, u1_in, u2_in, th_in, w_in, g1tmp_local, f3tmp_local, t)

    # tmp state at t+dt using k1
    in1_plus_a_in2(w_tp,  w_in,  k1_w,  dt)
    in1_plus_a_in2(th_tp, th_in, k1_th, dt)

    # compute u(t+dt) from w(t+dt)
    ComputeUVfromVort(w_tp, u1_tp, u2_tp)

    # step 2
    for i in range(local_n0):
        x = L*(local_start + i)/N0
        for j in range(N1):
            jj = i*N1 + j
            y = L*j/N1
            g1tmp_local[jj] = g1_func(t+dt, x, y, ConvTest, nu) + 0.0j
            f3tmp_local[jj] = f3_func(t+dt, x, y, ConvTest)     + 0.0j

    g1tmp_local = np.fft.fftn(g1tmp_local.reshape(local_n0, N1), norm=None).ravel()
    f3tmp_local = np.fft.fftn(f3tmp_local.reshape(local_n0, N1), norm=None).ravel()

    # k2
    RHS(k2_w, k2_th, u1_tp, u2_tp, th_tp, w_tp, g1tmp_local, f3tmp_local, t+dt)

    # update w, theta
    for i in range(local_n0):
        base = i*N1
        for j in range(N1):
            jj = base + j
            w_in[jj]  += dt * (k1_w[jj]  + k2_w[jj])  / 2.0
            th_in[jj] += dt * (k1_th[jj] + k2_th[jj]) / 2.0


def Read_VorTemp(N0=128, N1=128):
    data = np.loadtxt("initial_vorticity.txt")

    if data.size != N0 * N1:
        raise ValueError(f"File has {data.size} values but expected {N0*N1}")

    # flatten to 1D and cast to complex128
    w_all = data.astype(np.complex128).ravel()

    return w_all

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


def RHS(k1_w, k1_th, u1_in, u2_in, th_in, w_in, g1tmp_local, f3tmp_local, t):
    k = 11
def in1_plus_a_in2(w_tp,  w_in,  k1_w,  dt):
    k = 11

from mpi4py import MPI
import numpy as np


def RHS_k1_w_th(u1_in, u2_in, th_in, w_in, k1_w,  k1_th, t,dt, L):
    # Purpose: find k1_w =-\nabla\cdot(u w) + \nabla_x theta + g1
    # g1 = \nabla\times (f1,f2)
    # k1_th=-\nabla\cdot(u \theta) -u2 + f3
    # work in Fourier space
    g1tmp_local = np.empty(local_n0 * N1, dtype=np.complex128)
    f3tmp_local = np.empty(local_n0 * N1, dtype=np.complex128)

    # //g1tmp is external term of equation of w_t
    # //f3tmp is external term of equation of theta_t
    for i in range(local_n0):
        for j in range(N1):
            jj = i*N1+j
            x=L*(local_start+i)/N0
            y=L*j/N1
            g1tmp_local[jj] = g1_func(t, x, y,ConvTest, nu) + 0.0j
            f3tmp_local[jj] = f3_func(t+dt, x, y, ConvTest) + 0.0j

    g1tmp_local = np.fft.fftn(g1tmp_local.reshape(local_n0, N1), norm=None).ravel()
    f3tmp_local = np.fft.fftn(f3tmp_local.reshape(local_n0, N1), norm=None).ravel()

    RHS_BE(k1_w, k1_th, u1_in, u2_in, th_in, w_in, g1tmp_local, f3tmp_local, t)

def RHS_BE(k_w, k_th, in_u1, in_u2, in_th, in_w, g1tmp_local, f3tmp_local, time_spec):

    # compute RHS of ODE in Fourier space without diffusion terms
    # input: in_* for *=u1, u2, th (theta), pr (pressure)
    # output: RHS k_* for *=u1, u2, th (theta), pr (pressure)
    # k is F(y,t) in ODE dy/dt=F(y,t) without diffusion
    # both input and output are in Fourier mod
    # The following arrays denote temporary ones in computing N5 and N6:
    tmp1 = np.zeros(local_n0 * N1, dtype=np.complex128)
    tmp2 = np.zeros(local_n0 * N1, dtype=np.complex128)
    tmp3 = np.zeros(local_n0 * N1, dtype=np.complex128)
    tmp4 = np.zeros(local_n0 * N1, dtype=np.complex128)
    N4   = np.zeros(local_n0 * N1, dtype=np.complex128)
    N3   = np.zeros(local_n0 * N1, dtype=np.complex128)

    # Nonlinear terms N11, N12, N2, N3:
    #  Prepare for nonlinear calculation:
    N3, N4,in_u1, in_u2, in_w, in_th = PreNonlinear(in_u1, in_u2, in_w, in_th, N4, N3, time_spec);
    #  Step 1: Collect terms of (u\cdot\nabla)u to N11, N12 and N2:
    tmp1, tmp2, tmp3,tmp4 , N4, N3 =  Nonlinear3and4(in_u1, in_u2, in_w, in_th, tmp1, tmp2, tmp3, tmp4, N4, N3, time_spec);
    #  After all for nonlinear terms,  convert these quantities to Fourier space:
    in_u1, in_u2, in_w, in_th = AfterNonlinear(in_u1, in_u2, in_w, in_th, time_spec);

    #  Next, add linear terms:
    for i in range(local_n0):
           for j in range(N1):
               jj = i*N1+j #position in the 1-D local array
               k1 = wn[local_start + i]
               k2 = wn[j]
               k1sq =pow(k1,2)
               k2sq =pow(k2,2)
               ksq = k1sq + k2sq

              #  rhs of w: dw/dt = g1 -N4 + D_1 th without diffusion
               k_w[jj] = (g1tmp_local[jj] - N4[jj]) - 1j * k1 * in_th[jj]

              #  rhs of theta: dtheta/dt = f3 -N3 -u2  without diffusion
               k_th[jj] = f3tmp_local[jj] - N3[jj] - in_u2[jj]


def do_IMEX(u1_in,   u2_in, w_in,   th_in):

    # BE method on diffusion terms, explicit on other terms
    # u1n_in - u1_in )/dt = f
    # has 4 unknowns to solve: u1, u2, th (theta), pr (pressure)
    # work in Fourier space
    k1_w  = np.empty(alloc_local, dtype=np.complex128)
    k1_th = np.empty(alloc_local, dtype=np.complex128)
    # RHS of 3 equations:

    g1tmp_local = np.empty(alloc_local, dtype=np.complex128)
    f3tmp_local = np.empty(alloc_local, dtype=np.complex128)

    # (f1tmp, f2tmp) is external term of equation of u_t
    # f3tmp is external term of equation of theta_t
    for i in range(local_n0):
        for j in range(N1):
            jj = i * N1 + j
            x = L * (local_start + i) / N0
            y = L * j / N1
            g1tmp_local[jj] = g1_func(t + dt, x, y) + 0.0j
            f3tmp_local[jj] = f3_func(t + dt, x, y) + 0.0j

    g1tmp_local = np.fft.fftn(g1tmp_local.reshape(local_n0, N1), norm=None).ravel()
    f3tmp_local = np.fft.fftn(f3tmp_local.reshape(local_n0, N1), norm=None).ravel()

    # first, compute new u1, u2, and theta
    RHS_BE(k1_w, k1_th, u1_in, u2_in, th_in, w_in, g1tmp_local, f3tmp_local, t)

    for i in range(local_n0):
        for j in range(N1):
            jj = i*N1+j
            k1 = wn[local_start + i]
            k2 = wn[j]
            k1sq =pow(k1,2)
            k2sq =pow(k2,2)
            ksq = k1sq + k2sq

        # below is the new u1, u2, theta:
            tmp1 = 1.0 + dt*nu*k1sq
            tmp2 = 1.0 + dt*eta*k2sq

            w_in[jj]  = ( w_in[jj]  + dt*k1_w[jj] )/tmp1
            th_in[jj] = ( th_in[jj] + dt*k1_th[jj])/tmp2


def ComputeUVfromVort(w_in, u1_in, u2_in):
    # work in Fourier space
    # -Laplace psi = w, that is, \hat\psi = \hat{w}/|k|^2
    # u1 = D_y \psi,  or \hat{u1} = i*k2*\hat{\psi}  = i*k2/|k|^2 \hat{w}
    # u2 =-D_x \psi,  or \hat{u1} = -i*k1*\hat{\ps} =-i*k1/|k|^2 \hat{w}

    if( IN_FOURIER_SPACE[0]=='n'):
    #    DFT to convert data to Fourier space:
       w_in = np.fft.fftn(w_in.reshape(local_n0, N1), norm=None).ravel()


    if not np.iscomplexobj(u1_in):
        u1_in = u1_in.astype(np.complex128, copy=False)
    if not np.iscomplexobj(u2_in):
        u2_in = u2_in.astype(np.complex128, copy=False)

    for i in range(local_n0):
        for j in range(N1):
            jj = i*N1+j
            k1 = wn[local_start + i]
            k2 = wn[j]
            k1sq =pow(k1,2)
            k2sq =pow(k2,2)
            ksq = k1sq + k2sq

            if( ksq > 1.e-2 ):
                u1_in[jj] = (1j * k2 / ksq) * w_in[jj]
                u2_in[jj] = (-1j * k1 / ksq) * w_in[jj]

            else:
                u1_in[jj] = 0.0 + 0.0j
                u2_in[jj] = 0.0 + 0.0j

    if( IN_FOURIER_SPACE[0]=='n'):
    #    IDFT to convert data to physical space:
        u1_in = np.fft.ifftn(u1_in.reshape(local_n0, N1), norm="forward").real.ravel()
        u2_in = np.fft.ifftn(u2_in.reshape(local_n0, N1), norm="forward").real.ravel()
        w_in  = np.fft.ifftn(w_in.reshape(local_n0, N1),  norm="forward").real.ravel()

def do_BDF2(u1_in, u2_in, wo_in, w_in, wn_in, tho_in, th_in, thn_in, k1_w_old, k1_th_old):
    """
    Python port of the C do_BDF2:
      - Build real RHS in physical space (g1, f3)
      - Put them in complex buffers (imag=0)
      - Complex FFT to spectral space (same shape as inputs)
      - Call RHS_BE to fill k1_w, k1_th
      - BDF2 update for w and theta in spectral space
    Assumes all *_in arrays are 1D complex (length alloc_local), laid out row-major (i*N1 + j).
    """

    # --- allocate complex (not float) because we work in spectral space after FFT
    k1_w        = np.empty(alloc_local, dtype=np.complex128)
    k1_th       = np.empty(alloc_local, dtype=np.complex128)
    g1tmp_local = np.empty(alloc_local, dtype=np.complex128)
    f3tmp_local = np.empty(alloc_local, dtype=np.complex128)

    # --- build RHS in physical space (real), store into complex buffers (imag=0)
    for i in range(local_n0):
        x = L * (local_start + i) / N0
        for j in range(N1):
            y = L * j / N1
            jj = i * N1 + j
            # ensure real; keep imag=0
            g1tmp_local[jj] = float(g1_func(t, x, y, ConvTest, nu))
            f3tmp_local[jj] = float(f3_func(t, x, y, ConvTest))

    # --- complex FFT (same shape as inputs) — do NOT use rfftn or the size changes
    shape2d = (local_n0, N1)
    g1tmp_local = np.fft.fftn(g1tmp_local.reshape(shape2d), s=shape2d, norm=None).ravel()
    f3tmp_local = np.fft.fftn(f3tmp_local.reshape(shape2d), s=shape2d, norm=None).ravel()

    # If your C code applies a custom FFTW normalization, do it here to match:
    # fftw_normalize(g1tmp_local, DIM); fftw_normalize(f3tmp_local, DIM)

    # --- RHS_BE fills k1_w, k1_th in spectral space
    RHS_BE(k1_w, k1_th, u1_in, u2_in, th_in, w_in, g1tmp_local, f3tmp_local, t)

    # --- BDF2 update in spectral space
    for i in range(local_n0):
        k1 = wn[local_start + i]
        k1sq = k1 * k1
        for j in range(N1):
            jj = i * N1 + j
            k2 = wn[j]
            k2sq = k2 * k2

            # new w and theta (complex math ok; tmp1/tmp2 are real scalars)
            tmp1 = 1.5 / dt + nu * k1sq
            tmp2 = 1.5 / dt + eta * k2sq

            wn_in[jj]  = (2.0 * k1_w[jj]  - k1_w_old[jj]  + (2.0 * w_in[jj]  - 0.5 * wo_in[jj])  / dt) / tmp1
            thn_in[jj] = (2.0 * k1_th[jj] - k1_th_old[jj] + (2.0 * th_in[jj] - 0.5 * tho_in[jj]) / dt) / tmp2

    # --- roll states (complex arrays)
    k1_w_old[:] = k1_w
    k1_th_old[:] = k1_th

    wo_in[:]  = w_in
    w_in[:]   = wn_in
    tho_in[:] = th_in
    th_in[:]  = thn_in


# def do_BDF2(u1_in,u2_in, wo_in,   w_in, wn_in, tho_in,  th_in, thn_in, k1_w_old, k1_th_old):

#     # BDF2 method on linear terms, 2nd order extrapolation on nonlinear terms
#     # (3*u1n_in - 4*u1_in + 1*u1o_in)/(2dt) = f
#     # Add Shen and Yang 2010 's stability term
#     # has 3 unknowns to solve: u1, u2, th (theta), pr (pressure)
#     # work in Fourier space
#     # k1_pr_old is not used
#     k1_w        = np.empty(alloc_local, dtype=np.complex128)
#     k1_th       = np.empty(alloc_local, dtype=np.complex128)
#     # RHS of 3 equations:
#     g1tmp_local = np.empty(alloc_local, dtype=np.complex128)
#     f3tmp_local = np.empty(alloc_local, dtype=np.complex128)
#     # g1tmp is external term of equation of w_t
#     # f3tmp is external term of equation of theta_t
#     for i in range(local_n0):
#        for j in range(N1):
#           jj = i*N1+j
#           x=L*(local_start+i)/N0
#           y=L*j/N1
#           g1tmp_local[jj] = g1_func(t, x, y, ConvTest, nu)
#           f3tmp_local[jj] = f3_func(t, x, y, ConvTest)

#     g1tmp_local = np.fft.rfftn(g1tmp_local.reshape(local_n0, N1), norm=None).ravel()
#     f3tmp_local = np.fft.rfftn(f3tmp_local.reshape(local_n0, N1), norm=None).ravel()
#     # BDF2 should call RHS_BE
#     RHS_BE(k1_w, k1_th, u1_in, u2_in, th_in, w_in, g1tmp_local, f3tmp_local, t)

#     for i in range(local_n0):
#         for j in range(N1):
#             jj = i*N1+j
#             k1 = wn[local_start + i]
#             k2 = wn[j]
#             k1sq =pow(k1,2)
#             k2sq =pow(k2,2)
#             ksq = k1sq + k2sq

# 	#  below is the new u1, u2, theta:
#             tmp1 = 1.5/dt + nu*k1sq
#             tmp2 = 1.5/dt + eta*k2sq
#             wn_in[jj]  = (2 * k1_w[jj]  - k1_w_old[jj]  + (2 * w_in[jj]  - 0.5 * wo_in[jj]) / dt) / tmp1
#             thn_in[jj] = (2 * k1_th[jj] - k1_th_old[jj] + (2 * th_in[jj] - 0.5 * tho_in[jj]) / dt) / tmp2

#     k1_w_old[:] = k1_w
#     k1_th_old[:] = k1_th
#     wo_in[:]  = w_in
#     w_in[:]   = wn_in
#     tho_in[:] = th_in
#     th_in[:]  = thn_in


def in1_plus_a_in2(out, in1, in2, a):
    for i in range(local_n0):
        for j in range(N1):
            jj = i * N1 + j
            out[jj] = in1[jj] + a * in2[jj]

def assign(out, inp):
    out[:] = inp


def fftw_normalize(arr: np.ndarray, DIM: float):

    arr /= DIM

def filter_Krasny(arr: np.ndarray, noise_level: float):
    mask = np.abs(arr) < noise_level
    arr[mask] = 0.0 + 0.0j

def filter_exp(in1,Filter_alpha ):

    filter_alpha = Filter_alpha;      #NOTE: As filter_alpha increases, l2 (l0?) decays slower.

    for i in range(local_n0):
        exp_fx = math.exp(-36.0 * pow(2.0 * abs(wn[local_start + i]) / float(N0), Filter_alpha))
        for j in range(N1):
            exp_f = exp_fx * math.exp(-36.0 * pow(2.0 * abs(wn[j]) / float(N1), Filter_alpha))
            jj = i * N1 + j
            in1[jj] *= exp_f


def CompNormsInFourierSpace(u1_local, u2_local, th_local, u1, u2, th):
# compute norms in Fourier space
        PI = 3.14159265358979323846264338327950288419716939937510582097494459230781
        u_H2sq = 0.0
        th_H2sq = 0.0
        D1u_H2sq = 0.0
        D2th_H2sq = 0.0
        D1th_L2sq = 0.0
        tilde_th_H1sq = 0.0
        th_H1 = 0.0
        tilde_u_H1sq = 0.0
        D2_tilde_th_H2sq = 0.0
      #   Note: FFTW does not divide by N0*N1 after fotward DFT, so we have to divide it
        DIV= 0;
        if my_rank == 0:
            tilde_th_L2 = 0
            th_L2 = 0
            tilde_u_L2 = 0
            u_L2 = 0
            for i in range(N0):
               for j in range(N1):
                  jj = i*N1+j
                  k1 = wn[i]
                  k2 = wn[j]
                  k1sq =pow(k1,2)
                  k2sq =pow(k2,2)
                  ksq = k1sq + k2sq

                  # DFT of tilde{u}(x,y)= u(x,y) - \bar{u}(y) compute H1 norm
                  if( math.fabs(k1) >0.1):
                     tilde_u_H1sq  += (abs(u1[jj])**2) * (1 + k1sq + k2sq) + (abs(u2[jj])**2) * (1 + k1sq + k2sq)
                     tilde_th_H1sq += (abs(th[jj])**2) * (1 + k1sq + k2sq)
                     D2_tilde_th_H2sq += (abs(th[jj])**2) * (1 + k1sq + k2sq + k1sq**2 + k2sq**2 + k1sq*k2sq) * k2sq
                     tilde_th_L2 += abs(th[jj])**2
                     tilde_u_L2 += abs(u1[jj])**2 + abs(u2[jj])**2

                  th_L2 += abs(th[jj])**2
                  u_L2 += abs(u1[jj])**2 + abs(u2[jj])**2

                  tmp1 = (abs(u1[jj])**2 + abs(u2[jj])**2) * (1 + k1sq + k2sq + k1sq**2 + k2sq**2 + k1sq*k2sq)
                  u_H2sq += tmp1
                  D1u_H2sq += k1sq*tmp1

                  tmp2 = (abs(th[jj])**2) * (1 + k1sq + k2sq)
                  th_H1 += tmp2

                  tmp2 = (abs(th[jj])**2) * (1 + k1sq + k2sq + k1sq**2 + k2sq**2 + k1sq*k2sq)
                  th_H2sq   += tmp2
                  D2th_H2sq += k2sq * tmp2


                  tmp3 = k1sq * (abs(th[jj])**2)
                  D1th_L2sq += tmp3

                  DIV += abs(k1 * u1[jj] + k2 * u2[jj])**2

            # two scalars used in Parseval-Plancherel identity in integrals:
            tmp4 = 2*PI/DIM
            tmp5 = pow(tmp4,2)

            tilde_u_H1sq  *= tmp5
            tilde_th_H1sq *= tmp5
            D2_tilde_th_H2sq *= tmp5

            WuQ2 = math.sqrt(tilde_u_H1sq) + math.sqrt(tilde_th_H1sq) # First quantity in Theorem 2
            WuQ3 = t*( tilde_u_H1sq + tilde_th_H1sq ) # Second quantity in Theorem 2

            u_H2sq    *=tmp5
            th_H2sq   *=tmp5
            D1u_H2sq  *=tmp5
            D2th_H2sq *=tmp5
            D1th_L2sq *=tmp5
            th_H2 = math.sqrt(th_H2sq)		#H2 norm of theta
            th_H1 = tmp4*math.sqrt(th_H1)	#H1 norm of theta
            ThQ1 = tmp4*math.sqrt(th_L2)	#L2 norm of theta
            ThQ2 = tmp4*math.sqrt(tilde_th_L2)	#L2 norm of (theta - \bar{theta})
            ThQ3 = math.sqrt(tilde_th_H1sq)	#H1 norm of theta-bar{theta}
            UQ1 = tmp4*math.sqrt(u_L2)		#L2 norm of u
            UQ2 = tmp4*math.sqrt(tilde_u_L2)	#L2 norm of (u - \bar{u})
            UQ3  = math.sqrt(tilde_u_H1sq)	#H1 norm of u-bar{u}
            Integral1 = 0
            Integral2 = 0
            Integral3 = 0
            if(t <1e-10):
               Integral1 = 0
               Integral2 = 0
               Integral3 = 0
            else:
               Integral1 += 2*nu*dt*D1u_H2sq
               Integral2 += 2*eta*dt*D2_tilde_th_H2sq
               Integral3 += dt*D1th_L2sq

            WuQ1 = u_H2sq + th_H2sq + D1u_H2sq + D2th_H2sq + D1th_L2sq
            WuQ1 = u_H2sq + th_H2sq + Integral1 + Integral2 + Integral3


            DIV = math.sqrt(DIV)*tmp4

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
    cfg["DIM"] = cfg["N0"] * cfg["N1"]

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

def read_data(u1, u2, th, w,
              N0, N1, irestart):
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

def DefineWnWnabs(N0, N1, local_n0, local_start):

    # Build wavenumber vector
    wn = np.zeros(N0, dtype=float)
    for i in range(N0):
        if i <= N0 // 2:
            wn[i] = float(i)
        else:
            wn[i] = float(i - N0)

    # Build local |k| array
    wn_abs_local = np.zeros(local_n0 * N1, dtype=float)
    for i in range(local_n0):
        for j in range(N1):
            jj = i * N1 + j
            wn_abs_local[jj] = math.sqrt(wn[local_start + i]**2 + wn[j]**2)

    return wn, wn_abs_local

def MakeAverageZero(th_local, th_all):

# this average is the one in the whole domain


       if IN_FOURIER_SPACE[0]=='y':
          th_local = np.fft.irfftn(th_local.reshape(local_n0, N1), s=(local_n0, N1), norm="forward").ravel()

       if( my_rank == 0 ):
        #    compute average:
           ave = 0;
           for j in range(N1) :
               for i in range(N0):
                   jj=i*N1+j
                   ave += th_all[jj].real

           print("average = %12.5e\n", ave)
           ave = ave / (N0 * N1)
           th_all.real -= ave
      #  MPI_Scatter(th_all,alloc_local, MPI_DOUBLE_COMPLEX, th_local, alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD)

       if( IN_FOURIER_SPACE[0]=='y'):
          # convert to Fourier space:

            th_local = np.fft.rfftn(
                th_local.reshape(local_n0, N1),
                s=(local_n0, N1),
                norm=None
            ).ravel()

def ComputeThetaAverage(th_local, th_all, DIM):
       InitAverage = [0] * 100000
       if( IN_FOURIER_SPACE[0]=='y'):

        th_local = np.fft.fftn(th_local.reshape(local_n0, N1),s=(local_n0, N1),norm=None).ravel()
        fftw_normalize(th_local, DIM)
       if my_rank == 0 :
          #compute InitAverage:
          if(t < 1e-10 ):
             for j in range(N1):
                 tmp = 0
                 for i in range(N0):
                     jj=i*N1+j
                     tmp += th_all[jj].real
                 InitAverage[j] = tmp/N0


          #compute err:
          AveErrmaxtoinit = 0
          for j in range(N1):
              tmp = 0
              for i in range(N0):
                  jj=i*N1+j
                  tmp += th_all[jj].real

              tmp = math.fabs(tmp/N0- InitAverage[j])
              AveErrmaxtoinit = max(AveErrmaxtoinit, tmp)



       if( IN_FOURIER_SPACE[0]=='y'):
             th_local = np.fft.fftn(th_local.reshape(local_n0, N1),s=(local_n0, N1),norm=None).ravel()

def SetExactVort(w_local, time_spec):


      if( IN_FOURIER_SPACE[0]=='y'):
        #   convert to physical space:
                  #   convert to physical space:
        w_local = np.fft.irfftn(w_local.reshape(local_n0, N1),s=(local_n0, N1),norm=None).ravel()

        fftw_normalize(w_local)



      for i in range(local_n0):
        for  j in range(N1):
         jj=i*N1+j
         x=L*(local_start+i)/N0
         y=L*j/N1

         w_local[jj].real = winitFunc(time_spec, x,y)
         w_local[jj].image = 0




      if( IN_FOURIER_SPACE[0]=='y'):
        #   convert to physical space:
          w_local = np.fft.fftn(w_local.reshape(local_n0, N1),s=(local_n0, N1),norm=None).ravel()



def Output_Data(u1_local, u2_local, th_local, u1_all,   u2_all,   th_all,  w_local,  w_all):

      if( IN_FOURIER_SPACE[0]=='y'):
        # Convert to physical space (inverse FFT + normalization)
        u1_local = np.fft.irfftn(u1_local.reshape(local_n0, N1), s=(local_n0, N1), norm="None").ravel()
        u2_local = np.fft.irfftn(u2_local.reshape(local_n0, N1), s=(local_n0, N1), norm="None").ravel()
        th_local = np.fft.irfftn(th_local.reshape(local_n0, N1), s=(local_n0, N1), norm="None").ravel()
        w_local  = np.fft.irfftn(w_local.reshape(local_n0, N1),  s=(local_n0, N1), norm="None").ravel()

        fftw_normalize(u1_local, DIM)
        fftw_normalize(u2_local, DIM)
        fftw_normalize(th_local, DIM)
        fftw_normalize(w_local, DIM)

        if( my_rank == 0 ):
            printreal(u1_all, u2_all, th_all, w_all);
            FindError(u1_all, u2_all, th_all, w_all);
            printf("\n");


        if( IN_FOURIER_SPACE[0]=='y'):
          #   convert to Fourier space:


          u1_local = np.fft.rfftn(u1_local.reshape(local_n0, N1), s=(local_n0, N1), norm=None).ravel()
          u2_local = np.fft.rfftn(u2_local.reshape(local_n0, N1), s=(local_n0, N1), norm=None).ravel()
          th_local = np.fft.rfftn(th_local.reshape(local_n0, N1), s=(local_n0, N1), norm=None).ravel()
          w_local  = np.fft.rfftn(w_local.reshape(local_n0, N1),  s=(local_n0, N1), norm=None).ravel()

def InitVariables(w_local, th_local):
    if my_rank != 0:
        return
    for i in range(N0):
        x = L * i / N0
        base = i * N1
        for j in range(N1):
            y  = L * j / N1
            jj = base + j
            w_all[jj]  = complex(winitFunc(0.0, x, y), 0.0)
            th_all[jj] = complex(thinitFunc(0.0, x, y), 0.0)


def fftw_normalize(u, DIM):
    """Normalize like fftw_normalize in C."""
    return u / DIM

def CompCFLcondition(u1_local, u2_local, u1_all, u2_all,
                     IN_FOURIER_SPACE, N0, N1, dt, L,
                     comm=MPI.COMM_WORLD, my_rank=None):
    """
    Python equivalent of the C CompCFLcondition.
    Keeps same names and structure.
    """

    size = comm.Get_size()

    DIM = N0 * N1

    # --- convert to physical space if needed ---
    if IN_FOURIER_SPACE[0] == 'y':
        u1_local = np.fft.ifft2(u1_local.reshape(N0, N1)).real
        u2_local = np.fft.ifft2(u2_local.reshape(N0, N1)).real
        u1_local = fftw_normalize(u1_local, DIM)
        u2_local = fftw_normalize(u2_local, DIM)

    # --- gather data in physical space ---
    gathered_u1 = comm.gather(u1_local, root=0)
    gathered_u2 = comm.gather(u2_local, root=0)

    if my_rank == 0:
        if size > 1:
            u1_all[:] = np.vstack(gathered_u1).ravel()
            u2_all[:] = np.vstack(gathered_u2).ravel()
        else:
            u1_all[:] = u1_local.ravel()
            u2_all[:] = u2_local.ravel()

    # --- convert back to Fourier space if needed ---
    if IN_FOURIER_SPACE[0] == 'y':
        u1_local = np.fft.fft2(u1_local).ravel()
        u2_local = np.fft.fft2(u2_local).ravel()

    # --- compute umax on root ---
    umax = 0.0
    if my_rank == 0:
        for j in range(N1):
            for i in range(N0):
                jj = i * N1 + j
                tmp = u1_all[jj]**2 + u2_all[jj]**2
                umax = max(umax, tmp)
        umax = np.sqrt(umax)
    umax = comm.bcast(umax, root=0)

    # --- CFL condition check ---
    h = L / N0
    tmp = dt * umax / h
    CFL_break = 0
    if tmp > 0.5:
        CFL_break = 1
        if my_rank == 0:
            print("***********************************")
            print(f"    dt        = {dt:12.5e}")
            print(f"    umax      = {umax:12.5e}")
            print(f"    h         = {h:12.5e}")
            print(f"CFL=dt*umax/h = {tmp:12.5e}")
            print("CFL is too big, need to decrease dt")
            print("***********************************")

    CFL_break = comm.bcast(CFL_break, root=0)

    return CFL_break, umax, h, u1_local, u2_local, u1_all, u2_all

def test_909(w):


    w[0] += 200
    w[1] += 400


def InitFFTW(wo_local, w_local, tho_local, th_local,
             u1o_local, u1_local, u2o_local, u2_local):
    """
    Python equivalent of the C InitFFTW.
    Converts all given fields from physical space to Fourier space.
    Arrays are assumed to be 1D views of (local_n0, N1) data.
    """
    # Reshape to 2D (local_n0, N1), FFT, then flatten back
    wo_local[:]  = np.fft.fftn(wo_local.reshape(local_n0, N1)).ravel()
    w_local[:]   = np.fft.fftn(w_local.reshape(local_n0, N1)).ravel()
    tho_local[:] = np.fft.fftn(tho_local.reshape(local_n0, N1)).ravel()
    th_local[:]  = np.fft.fftn(th_local.reshape(local_n0, N1)).ravel()
    u1o_local[:] = np.fft.fftn(u1o_local.reshape(local_n0, N1)).ravel()
    u1_local[:]  = np.fft.fftn(u1_local.reshape(local_n0, N1)).ravel()
    u2o_local[:] = np.fft.fftn(u2o_local.reshape(local_n0, N1)).ravel()
    u2_local[:]  = np.fft.fftn(u2_local.reshape(local_n0, N1)).ravel()


import math
from mpi4py import MPI


N0 = N1 = DIM = None
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

my_rank = None
nprocs = None
alloc_local = None
local_n0 = None
local_start = None

IN_FOURIER_SPACE = ['n']   # mutable so helpers can flip between 'n'/'y'
wn = None
wn_abs_local = None

# running-time scalars used across helpers
t = 0.0
iout = 0
iter_print = 0
iter_norms = 0
CFL_break = 0

# “norms” outputs used in your printing
Integral1 = Integral2 = Integral3 = 0.0
WuQ1 = WuQ2 = WuQ3 = 0.0
ThQ1 = ThQ2 = ThQ3 = 0.0
UQ1 = UQ2 = UQ3 = 0.0
DIV = 0.0
umax = 0.0
AveErrmaxtoinit = 0.0

# -------------- put your existing helpers above this line --------------
# (do_IMEX, do_BDF2, do_RK2, ComputeUVfromVort, CompNormsInFourierSpace,
#  ComputeThetaAverage, MakeAverageZero, Output_Data, DefineWnWnabs, read_input,
#  read_data, Read_VorTemp, filter_exp, filter_Krasny, assign, in1_plus_a_in2, etc.)

def _split_rows(total_rows: int, size: int, rank: int):
    """Even block row decomposition like FFTW MPI would do."""
    base = total_rows // size
    extra = total_rows % size
    n0 = base + (1 if rank < extra else 0)
    start = rank * base + min(rank, extra)
    return n0, start

def main():
    global N0, N1, DIM, METHOD, TMAX, dt, dt_print, dt_norms, eta, nu, WuEpsi
    global        restart, irestart, ConvTest, ShenYang, USE_Filter, Filter_alpha
    global     Filter_noiselevel, my_rank, nprocs, alloc_local, local_n0
    global       local_start, IN_FOURIER_SPACE, wn, wn_abs_local, t, iout
    global    iter_print, iter_norms, CFL_break, Integral1, Integral2, Integral3
    global        WuQ1, WuQ2, WuQ3, ThQ1, ThQ2, ThQ3, UQ1, UQ2, UQ3, DIV, umax
    global       AveErrmaxtoinit

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if my_rank == 0:
        cfg = read_input("input.ini")
    else:
        cfg = None

    # broadcast config
    cfg = comm.bcast(cfg, root=0)

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
    DIM              = cfg['DIM']

    CFL_break = 0
    max_iter = int(math.ceil(TMAX / dt))
    iter_print = int(math.ceil(dt_print / dt))
    iter_norms = int(math.ceil(dt_norms / dt))

    if my_rank == 0:
        print(f"TMAX, MAX_ITER = {TMAX:.6f}, {max_iter:.6f}")
        print(f"dt_print = {dt_print:.6f}, iter_print={iter_print}")
        print(f"dt_norms = {dt_norms:.6f}, iter_norms={iter_norms}")
        print(f"dt = {dt:.6e}")
        print(f"N0, N1 = {N0}, {N1}")
        print(f"hx, hy = {L/N0:12.5e}, {L/N1:12.5e}")

    local_n0, local_start = _split_rows(N0, nprocs, my_rank)
    alloc_local           = local_n0 * N1  # # of complex entries on this rank

    if my_rank == 0:
        print(f"my_rank = {my_rank}")
        print(f"local_n0    = {local_n0}")
        print(f"local_start = {local_start}")
        print(f"alloc_local = {alloc_local}")

    # local fields (complex128)
# if these will hold Fourier-space values at any point, make them complex
    u1_local  = np.zeros(alloc_local, dtype=np.complex128)
    u1o_local = np.zeros(alloc_local, dtype=np.complex128)
    u2_local  = np.zeros(alloc_local, dtype=np.complex128)
    u2o_local = np.zeros(alloc_local, dtype=np.complex128)
    th_local  = np.zeros(alloc_local, dtype=np.complex128)
    tho_local = np.zeros(alloc_local, dtype=np.complex128)
    thn_local = np.zeros(alloc_local, dtype=np.complex128)
    w_all = np.empty(DIM, dtype=np.complex128)
    w_local   = np.zeros(alloc_local, dtype=np.complex128)
    wo_local  = np.zeros(alloc_local, dtype=np.complex128)
    wn_local  = np.zeros(alloc_local, dtype=np.complex128)  # if you store Fourier wn

    k1_w_old  = np.zeros(alloc_local, dtype=np.complex128)
    k1_th_old = np.zeros(alloc_local, dtype=np.complex128)

    th_all = np.zeros(DIM, dtype=np.complex128)  # not float
    u1_all = np.zeros(DIM, dtype=np.complex128)
    u2_all = np.zeros(DIM, dtype=np.complex128)
    wo_all = np.zeros(DIM, dtype=float) if my_rank == 0 else np.empty(0, dtype=float)

    wn, wn_abs_local = DefineWnWnabs(N0, N1, local_n0, local_start)
    t = 0.0
    iout = 0
    iter_start = 0

    if restart[0].lower() == 'n':
        # root prepares initial w, theta
        if my_rank == 0:
            # w_all = Read_VorTemp(N0, N1)
            InitVariables(w_local, th_local)
        comm.Bcast([w_all,  MPI.COMPLEX16], root=0)
        comm.Bcast([th_all, MPI.COMPLEX16], root=0)
        # scatter to ranks (equal blocks of length alloc_local)
        comm.Scatter([w_all,  MPI.COMPLEX16], [w_local,  MPI.COMPLEX16], root=0)
        comm.Scatter([th_all, MPI.COMPLEX16], [th_local, MPI.COMPLEX16], root=0)

        IN_FOURIER_SPACE[0] = 'n'
        ComputeUVfromVort(w_local, u1_local, u2_local)

        MakeAverageZero(th_local, th_all)
        MakeAverageZero(u1_local, u1_all)
        MakeAverageZero(u2_local, u2_all)

        IN_FOURIER_SPACE[0] = 'n'
        ComputeUVfromVort(w_local, u1_local, u2_local)  # u1_local,u2_local at t=0

        Output_Data(u1_local, u2_local, th_local, u1_all, u2_all, th_all, w_local, w_all)


    elif restart[0].lower() == 'y':
        if my_rank == 0:
            # read into global arrays on root
            t, Integral1, Integral2, Integral3 = read_data(u1_all, u2_all, th_all, w_all, N0, N1, irestart)
        # broadcast current time (and integrals if you use them elsewhere)
        t = comm.bcast(t, root=0)

        iout        = int(math.ceil(t / dt_print))
        iter_start  = int(math.ceil(t / dt))

        if my_rank == 0:
            print("restart info *********************")
            print(f"restart      = {restart}")
            print(f"restart time = {t:.6e}")
            print(f"iout         = {iout}")
            print(f"iter_start   = {iter_start}")

        comm.Scatter([w_all,  MPI.COMPLEX16], [w_local,  MPI.COMPLEX16], root=0)
        comm.Scatter([th_all, MPI.COMPLEX16], [th_local, MPI.COMPLEX16], root=0)

        IN_FOURIER_SPACE[0] = 'n'
        ComputeUVfromVort(w_local, u1_local, u2_local)  # at time t
    else:
        if my_rank == 0:
            print("This input for restart is invalid")
        MPI.Finalize()
        return

    #  go to Fourier space
    InitFFTW(wo_local, w_local, tho_local, th_local, u1o_local, u1_local, u2o_local, u2_local)
    IN_FOURIER_SPACE[0] = 'y'

    #  test initial norms
    CompNormsInFourierSpace(u1_local, u2_local, th_local, u1_all, u2_all, th_all)
    ComputeThetaAverage(th_local, th_all, DIM)

    if my_rank == 0:
        with open("norms", "a") as f:
            f.write(f"{t:12.5e}   {WuQ1:12.5e}   {WuQ2:12.5e}   {WuQ3:12.5e}   "
                    f"{DIV:12.5e}  {umax:12.5e}  {AveErrmaxtoinit:12.5e}\n")
        with open("morenorms", "a") as f:
            f.write(f"{t:12.5e}   {ThQ1:12.5e}   {ThQ2:12.5e}   {ThQ3:12.5e}   "
                    f"{UQ1:12.5e}  {UQ2:12.5e}  {UQ3:12.5e}\n")
            print(f"ThQ1={ThQ1:12.5e},  ThQ2={ThQ2:12.5e}")

    #  main time loop
    for it in range(iter_start, max_iter):
        # CFL check
        shape = N0*N1

        CFL_break, umax, h, u1_local, u2_local, u1_all, u2_all = CompCFLcondition(
          u1_local, u2_local, u1_all, u2_all,
          IN_FOURIER_SPACE, N0, N1, dt, L,
          comm, my_rank
        )
        if CFL_break == 1:
            Output_Data(u1_local, u2_local, th_local, u1_all, u2_all, th_all, w_local, w_all)
            if my_rank == 0:
                # Mimic MPI_Abort + exit(1)
                print("CFL violated; aborting.")
            MPI.COMM_WORLD.Abort(1)
            return

        # time stepper
        if METHOD == 1:  # BE/IMEX
            do_IMEX(u1_local, u2_local, w_local, th_local)
            ComputeUVfromVort(w_local, u1_local, u2_local)

        elif METHOD == 2:  # BDF2
            if it == iter_start:
                assign(wo_local,  w_local)
                assign(tho_local, th_local)
                assign(u1o_local, u1_local)
                assign(u2o_local, u2_local)

                RHS_k1_w_th(u1o_local, u2o_local, tho_local, wo_local, k1_w_old, k1_th_old, dt,t, L)
                do_RK2(u1_local, u2_local, w_local, th_local)
                ComputeUVfromVort(w_local, u1_local, u2_local)
            else:
                do_BDF2(u1_local, u2_local,
                        wo_local, w_local, wn_local,
                        tho_local, th_local, thn_local,
                        k1_w_old, k1_th_old)
                ComputeUVfromVort(wn_local, u1_local, u2_local)

        elif METHOD == 3:  # RK2
            do_RK2(u1_local, u2_local, w_local, th_local)
            ComputeUVfromVort(w_local, u1_local, u2_local)

        # advance time
        t += dt
        if USE_Filter == 1:
            filter_exp(u1_local, Filter_alpha)
            filter_exp(u2_local, Filter_alpha)
            filter_exp(th_local, Filter_alpha)
            filter_exp(w_local,  Filter_alpha)
        elif USE_Filter == 2:
            filter_Krasny(u1_local, Filter_noiselevel)
            filter_Krasny(u2_local, Filter_noiselevel)
            filter_Krasny(th_local, Filter_noiselevel)
            filter_Krasny(w_local,  Filter_noiselevel)

        # norms
        CompNormsInFourierSpace(u1_local, u2_local, th_local, u1_all, u2_all, th_all)
        ComputeThetaAverage(th_local, th_all, DIM)

        if (it + 1) % iter_norms == 0 and my_rank == 0:
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
            if my_rank == 0:
                print(f"iter, time = {it+1}, {t}")
            Output_Data(u1_local, u2_local, th_local, u1_all, u2_all, th_all, w_local, w_all)

    #  finalize
    MPI.Finalize()

if __name__ == "__main__":
    main()
