r"""
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
   N5 = - (D_t - η D_22) P(u·∇u) + ∇^⊥ D_1 Laplace⁻¹(u·∇θ)
   N6 = - (D_t - ν D_11)(u·∇θ) + u·∇u₂ - D_2 Laplace⁻¹ ∇(u·∇u)

where P is Leray's projection, that is,
   P(u) = u - \nabla \Laplace^{-1} (\nabla\cdot u)

Method: convert to first order differential system on t: introducing new variables
   v1  = D_t u1
   v2  = D_t u2
   ps = D_t theta
"""

def g1_func(t, x, y,ConvTest, nu):

    if(ConvTest):
        # //w=  -cos(t)*2*a*cos(2*a*x)*pow(sin(a*y),2)  -cos(t)*pow(sin(a*x),2)*2*a*cos(2*a*y)
        # //theta= cos(t)*pow(sin(a*x),2)*sin(2*a*y)

        u1   = math.cos(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)
        u2   =-math.cos(t)*math.sin(2*A*x)*pow(math.sin(A*y),2)
        Dtw  = math.cos(t)*2*A*math.cos(2*A*x)*pow(math.sin(A*y),2) +math.sin(t)*pow(math.sin(A*x),2)*2*A*math.cos(2*A*y)
        D1w  = math.cos(t)*4*A*A*math.sin(2*A*x)*pow(math.sin(A*y),2)-math.cos(t)*math.sin(2*A*x)*2*A*A*math.cos(2*A*y)
        D2w  =-math.cos(t)*2*A*A*math.cos(2*A*x)*math.sin(2*A*y) + math.cos(t)*pow(math.sin(A*x),2)*4*A*A*math.sin(2*A*y)
        D11w = math.cos(t)*8*A*A*A*math.cos(2*A*x)*pow(math.sin(A*y),2) -math.cos(t)*math.cos(2*A*x)*4*A*A*A*math.sin(2*A*y)
        D1th = math.cos(t)*A*math.sin(2*A*x)*math.sin(2*A*y)

        return Dtw + u1*D1w + u2*D2w - D1th - nu*D11w
    else:
        return 0

def f3_func( t,  x,  y, ConvTest):

    if( ConvTest):

        u1   = math.cos(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)
        u2   =-math.cos(t)*math.sin(2*A*x)*pow(math.sin(A*y),2)
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
     return math.cos(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)
  else:
     return WuEpsi*math.cos(t)*pow(math.sin(A*x),2)*math.sin(2*A*y)

def u2initFunc( t,  x,  y):
  if(ConvTest):
     return -math.cos(t)*math.sin(2*A*x)*pow(math.sin(A*y),2)
  else:
     return -WuEpsi*math.cos(t)*math.sin(2*A*x)*pow(math.sin(A*y),2)

def AfterNonlinear( u1, u2, w, th, time_spec):

    u1 = np.fft.fftn(u1, norm="backward")
    u2 = np.fft.fftn(u2, norm="backward")
    th = np.fft.fftn(th, norm="backward")
    w  = np.fft.fftn(w,  norm="backward")

    return u1, u2, w, th

def PreNonlinear(u1, u2, w, th,N3, N4 , time_spec):

    # Above all, convert input quantities to Physical space:
    u1 = np.fft.ifftn(u1,norm="backward").real
    u2 = np.fft.ifftn(u2,norm="backward").real
    th = np.fft.ifftn(th,norm="backward").real
    w = np.fft.ifftn(w,norm="backward").real
    # divide them by DIM because FFTW does not perform it:
    # initialize N5 and N6:
    N3[:] = 0.0 + 0.0j
    N4[:] = 0.0 + 0.0j

    return N3, N4,u1, u2, w, th
def Nonlinear3and4(u1, u2, w, th, tmp1,  tmp2,  tmp3, tmp4, N4,  N3, time_spec):

    # computer N3=- (u\cdot\nabla)\theta
    # computer N4=- (u\cdot\nabla) w
    # Denote z = hat (u\cdot nabla)\theta = hat( div(u*theta) ):
    #          = hat( D_1(u1*theta) + D_2(u2*theta) )


    # first we need u1*theta, u1*theta in Physical space
    for i in range(N0):
        for j in range(N1):
            jj = i*N1+j;
#    set u1 and u2 as exact values to check on p:

    jj=i*N1+j
    x=L*(i)/N0
    y=L*j/N1

    tmp1[jj] = u1[jj].real * th[jj].real + 0.0j  # u1*theta, real only
    tmp2[jj] = u2[jj].real * th[jj].real + 0.0j  # u2*theta
    tmp3[jj] = u1[jj].real * w[jj].real   + 0.0j # u1*vorticity
    tmp4[jj] = u2[jj].real * w[jj].real   + 0.0j # u2*vorticity

#    second we compute the DFT of u1*theta, u2*theta:
    tmp1 = np.fft.fftn(tmp1, axes=(0,1), norm="backward")
    tmp2 = np.fft.fftn(tmp2, axes=(0,1),  norm="backward")
    tmp3 = np.fft.fftn(tmp3, axes=(0,1),  norm="backward").ravel()
    tmp4 = np.fft.fftn(tmp4, axes=(0,1),  norm="backward")
#    third, compute N3 and N4:
    for i in range(N0):
        for j in range(N1):
            jj = i*N1+j;                    # position in the 1-D local array
            k1 = wn[ i]
            k2 = wn[j]
            ksq = pow(k1,2) + pow(k2,2)

            # N3 = z = hat( D_1(u1*theta) + D_2(u2*theta) )
            z = (-k1*tmp1[jj].imag - k2*tmp2[jj].imag)  +  (k1*tmp1[jj].real + k2*tmp2[jj].real)
            N3[jj] += z


            # N4 = z = hat( D_1(u1*vort) + D_2(u2*vort) )
            z = (-k1*tmp3[jj].imag - k2*tmp4[jj].imag) +  (k1*tmp3[jj].real + k2*tmp4[jj].real)
            N4[jj]= z

    return tmp1, tmp2, tmp3, tmp4, N3, N4

def do_RK2(u1, u2, w, th,k1_w, k1_th, k2_w, k2_th,w_tp, th_tp, u1_tp, u2_tp,g1tmp, f3tmp, t, dt, L, ConvTest, wn, N3, N4, tmp1, tmp2, tmp3, tmp4):
#     # // RK2 method
#     # // has 2 unknowns to solve: w (vorticity) and th (theta)
#     # // work in Fourier space
    x = (np.arange(N0) * L / N0)[:, None]   # (N0, 1)
    y = (np.arange(N1) * L / N1)[None, :]   # (1, N1)
    g1tmp = g1_func(t + dt, x, y, ConvTest, nu).astype(np.complex128)
    f3tmp = f3_func(t + dt, x, y, ConvTest).astype(np.complex128)
    
    g1tmp = np.fft.fftn(g1tmp, norm="backward")
    f3tmp = np.fft.fftn(f3tmp, norm="backward")


    # k1
    k1_w, k1_th = RHS(k1_w, k1_th, u1,u2, th, w,
                      g1tmp, f3tmp, t,
                      wn, N3,N4, tmp1, tmp2, tmp3, tmp4)

    # tmp state at t+dt using k1
    in1_plus_a_in2(w_tp,  w,  k1_w,  dt)
    in1_plus_a_in2(th_tp, th, k1_th, dt)

    # compute u(t+dt) from w(t+dt)
    w_tp, u1_tp, u2_tp = ComputeUVfromVort(w_tp, u1_tp, u2_tp)

    # step 2
    for i in range(N0):
        x = L*( i)/N0
        for j in range(N1):
            jj = i*N1 + j
            y = L*j/N1
            g1tmp[jj] = np.complex128(g1_func(t, x, y, ConvTest, nu))
            f3tmp[jj] = np.complex128(f3_func(t, x, y, ConvTest))
    g1tmp = np.fft.fftn(
        g1tmp,
        norm="backward"
    ).real.ravel()

    # Inverse FFT
    f3tmp = np.fft.ifftn(
        f3tmp,
        norm="backward"
    ).real.ravel()

    # k2
    k2_w, k2_th =RHS(k2_w, k2_th, u1, u2, th, w,
                    g1tmp, f3tmp, t,
                     wn, N3, N4, tmp1, tmp2, tmp3, tmp4)

    # update w, theta
    for i in range(N0):
        base = i*N1
        for j in range(N1):
            jj = base + j
            w[jj]  += dt * (k1_w[jj]  + k2_w[jj])  / 2.0
            th[jj] += dt * (k1_th[jj] + k2_th[jj]) / 2.0

    return w, th
def Read_VorTemp():
    data = np.loadtxt("initial_vorticity.txt")

    if data.size != N0 * N1:
        raise ValueError(f"File has {data.size} values but expected {N0*N1}")

    # flatten to 1D and cast to complex128
    w = data.astype(np.complex128).ravel()

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



def RHS_k1_w_th(u1,u2,th, w, k1_w,  k1_th, t,dt, g1tmp, f3tmp, N3,N4,tmp1, tmp2, tmp3, tmp4):
    # Purpose: find k1_w =-\nabla\cdot(u w) + \nabla_x theta + g1
    # g1 = \nabla\times (f1,f2)
    # k1_th=-\nabla\cdot(u \theta) -u2 + f3
    # work in Fourier space

    # g1tmp is external term of equation of w_t
    # f3tmp is external term of equation of theta_t
    x = (np.arange(N0) * L / N0)[:, None]  
    y = (np.arange(N1) * L / N1)[None, :]  
    g1tmp = g1_func(t, x, y, ConvTest, nu) + 0j 
    f3tmp = f3_func(t, x, y, ConvTest) + 0j  

    g1tmp = np.fft.fftn(g1tmp, axes=(0, 1), norm="backward")
    f3tmp = np.fft.fftn(f3tmp, axes=(0, 1), norm="backward")

    # k1 is F(t_n, u_n) of ODE u_t=F(t,u) without diffusion terms:
    k1_w, k1_th = RHS_BE(k1_w, k1_th,u1, u2, th, w, g1tmp, f3tmp, t,wn, N3, N4,tmp1, tmp2, tmp3, tmp4)
    return k1_w, k1_th

def RHS_BE(k_w, k_th, u1, u2, th, w,
           g1tmp, f3tmp, t,
           wn, N3, N4,tmp1, tmp2, tmp3, tmp4):

    # compute RHS of ODE in Fourier space without diffusion terms
    # input: in_* for *=u1, u2, th (theta), pr (pressure)
    # output: RHS k_* for *=u1, u2, th (theta), pr (pressure)
    # k is F(y,t) in ODE dy/dt=F(y,t) without diffusion
    # both input and output are in Fourier mod
    # The following arrays denote temporary ones in computing N5 and N6:

    # Nonlinear terms N11, N12, N2, N3:
    #  Prepare for nonlinear calculation:
    N3, N4, u1, u2, w, th = PreNonlinear(u1, u2, w, th, N3,N4, t)
    #  Step 1: Collect terms of (u\cdot\nabla)u to N11, N12 and N2:
    tmp1, tmp2, tmp3,tmp4 , N3,N4 =  Nonlinear3and4(u1, u2, w, th, tmp1, tmp2, tmp3, tmp4, N3,N4, t)
    #  After all for nonlinear terms,  convert these quantities to Fourier space:
    u1, u2, w, th = AfterNonlinear(u1, u2, w, th, t)

    k1 = wn[:, None]        # shape (N0, 1)
    k2 = wn[None, :]        # shape (1, N1)
    # Next, add linear terms:
    # rhs of w: dw/dt = g1 - N4 - i*k1*theta
    k_w[:, :] = (g1tmp - N4) - 1j * k1 * th
    # rhs of theta: dtheta/dt = f3 - N3 - u2
    k_th[:, :] = f3tmp - N3 - u2
    return k_w, k_th

def do_IMEX(k1_w, k1_th, u1, u2, th, w, g1tmp, f3tmp, t, wn, N3,N4,tmp1, tmp2, tmp3, tmp4):

    # BE method on diffusion terms, explicit on other terms
    # u1n_in - in_u1 )/dt = f
    # has 4 unknowns to solve: u1, u2, th (theta), pr (pressure)
    # work in Fourier space
    # (f1tmp, f2tmp) is external term of equation of u_t
    # f3tmp is external term of equation of theta_t
    x = (np.arange(N0) * L / N0)[:, None]   # shape (N0, 1)
    y = (np.arange(N1) * L / N1)[None, :]   # shape (1, N1)
    g1tmp[:, :] = g1_func(t, x, y, ConvTest, nu)
    f3tmp[:, :] = f3_func(t, x, y, ConvTest)

    # Forward FFT
    g1tmp = np.fft.fftn(g1tmp, axes=(0, 1), norm="backward")
    f3tmp = np.fft.fftn(f3tmp, axes=(0, 1), norm="backward")
    # first, compute new u1, u2, and theta
    k1_w, k1_th = RHS_BE(k1_w, k1_th, u1, u2, th, w, g1tmp, f3tmp, t, wn, N3,N4,tmp1, tmp2, tmp3, tmp4)

    #  Diffusion terms
    k1 = wn[:, None]   
    k2 = wn[None, :] 
    k1sq = k1**2
    k2sq = k2**2

    denom_w  = 1.0 + dt *nu* k1sq
    denom_th = 1.0 + dt *eta* k2sq

    w  = (w  + dt * k1_w)  / denom_w
    th = (th + dt * k1_th) / denom_th

    return w, th

def ComputeUVfromVort(w, u1, u2):
    r"""
    work in Fourier space
    -Laplace psi = w, that is, \hat\psi = \hat{w}/|k|^2
    u1 = D_y \psi,  or \hat{u1} = i*k2*\hat{\psi}  = i*k2/|k|^2 \hat{w}
    u2 =-D_x \psi,  or \hat{u1} = -i*k1*\hat{\ps} =-i*k1/|k|^2 \hat{w}
    """
    # If in physical space, move to Fourier for algebra:
    if IN_FOURIER_SPACE[0] == 'n':
        w = np.fft.fftn(w, norm="backward")
    kx = wn[:, None]   # shape (N0, 1)
    ky = wn[None, :]   # shape (1, N1)
    ksq = kx**2 + ky**2
    mask = ksq > 1e-12
    if not np.iscomplexobj(u1):
        u1[:] = u1.astype(np.complex128, copy=False)
    if not np.iscomplexobj(u2):
        u2[:] = u2.astype(np.complex128, copy=False)

    if not np.iscomplexobj(u1):
        u1 = u1.astype(np.complex128, copy=False)
    if not np.iscomplexobj(u2):
        u2 = u2.astype(np.complex128, copy=False)

    u1[:] = np.where(mask, (1j * ky / ksq) * w, 0.0)
    u2[:] = np.where(mask, (-1j * kx / ksq) * w, 0.0)

    if IN_FOURIER_SPACE[0] == 'n':
        u1 = np.fft.ifftn(u1, norm="backward").real
        u2 = np.fft.ifftn(u2, norm="backward").real
        w  = np.fft.ifftn(w,  norm="backward").real

    return w, u1, u2

def do_BDF2(u1,u2, wo, w, wn,
            tho, th, thn,
            k1_w_old, k1_th_old,
            k1_w, k1_th,
            tmp1, tmp2, tmp3, tmp4, g1tmp, f3tmp,N3,N4):
    """
    Build RHS in physical space (g1, f3) and them in complex buffers (imag=0)
    Call RHS_BE to fill k1_w, k1_th
    BDF2 update for w and theta in spectral space
    all *_in arrays are 1D complex
    do Backward Differentiation Formula of order 2
    """
    x = (np.arange(N0) * L / N0)[:, None]  
    y = (np.arange(N1) * L / N1)[None, :]  

    g1tmp = g1_func(t, x, y, ConvTest, nu) + 0j
    f3tmp = f3_func(t, x, y, ConvTest) + 0j
    g1tmp = np.fft.fftn(g1tmp, norm="backward").ravel()
    f3tmp = np.fft.fftn(f3tmp, norm="backward").ravel()
    
    k1_w, k1_th = RHS_BE(k1_w, k1_th,u1, u2, th, w,g1tmp, f3tmp, t, wn, N3, N4,tmp1, tmp2, tmp3, tmp4)

    k1sq = wn[:, None]
    k2sq = wn[None, :]

    denom_w  = 1.5 / dt + nu * (k1sq**2)
    denom_th = 1.5 / dt + eta * (k2sq**2)

    wn  = (2.0 * k1_w - k1_w_old + (2.0 * w - 0.5 * wo) / dt) / denom_w
    thn = (2.0 * k1_th - k1_th_old + (2.0 * th - 0.5 * tho) / dt) / denom_th

    return k1_w, k1_th  ,wn,thn

def in1_plus_a_in2(out, in1, in2, dt):
    out[:] = in1 + A * in2
    return out

def filter_Krasny(arr: np.ndarray, noise_level: float):
    mask = np.abs(arr) < noise_level
    arr[mask] = 0.0 + 0.0j
    return arr

def filter_exp(in1,Filter_alpha ):
    kx = wn[:, None]   # shape (N0, 1)
    ky = wn[None, :]   # shape (1, N1)


    exp_fx = np.exp(-36.0 * (2.0 * np.abs(kx) / N0) ** Filter_alpha)
    exp_fy = np.exp(-36.0 * (2.0 * np.abs(ky) / N1) ** Filter_alpha)

    exp_filter = exp_fx * exp_fy
    in1[:] *= exp_filter
    return in1


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
    mask = np.abs(k1) > 1e-12
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

    # Build wavenumber vector
    for i in range(len(wn)):
        if i <= len(wn) // 2:
            wn[i] = float(i)
        else:
            wn[i] = float(i - len(wn))

    # Full 2D |k| array (flattened)
    wn_abs_local = np.zeros(N0 * N1, dtype=float)
    for i in range(N0):
        for j in range(N1):
            jj = i * N1 + j
            kx = wn[i]
            ky = wn[j]
            wn_abs_local[jj] = math.sqrt(kx**2 + ky**2)

    return wn, wn_abs_local


def ComputeThetaAverage(th):
      if IN_FOURIER_SPACE[0] == 'y':
        th = np.fft.irfftn(th, norm="backward")

      #compute InitAverage:
      if t < 1e-10:
        InitAverage = th.real.mean(axis=0) 

      tmp = th.real.mean(axis=0)     
      ff = tmp - InitAverage
      AveErrmaxtoinit = tmp.max()

      if IN_FOURIER_SPACE[0] == 'y':
        th = np.fft.fftn(th, norm="backward")

      return AveErrmaxtoinit, th

def SetExactVort(w, time_spec):

      if( IN_FOURIER_SPACE[0]=='y'):
        #   convert to physical space:
        w = np.fft.irfftn(w,norm="backward").real()

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
        u1 = np.fft.irfftn(u1, norm="backward")
        u2 = np.fft.irfftn(u2, norm="backward")
        th = np.fft.irfftn(th, norm="backward")
        w  = np.fft.irfftn(w,  norm="backward")

        printreal(u1, u2, th, w,t, Integral1, Integral2, Integral3, iout, CFL_break, N0, N1);
        FindError(u1, u2, th, w)

      if( IN_FOURIER_SPACE[0]=='y'):
        #   convert to Fourier space:
        u1 = np.fft.fftn(u1, norm="backward")
        u2 = np.fft.fftn(u2, norm="backward")
        th = np.fft.fftn(th, norm="backward")
        w  = np.fft.fftn(w,  norm="backward")
      return u1, u2, th, w

def InitVariables(w, th):

    x = (np.arange(N0) * L / N0)[:, None]   
    y = (np.arange(N1) * L / N1)[None, :] 

    w[:, :] = winitFunc(0.0, x, y) + 0j
    th[:, :] = thinitFunc(0.0, x, y) + 0j

def CompCFLcondition( u1, u2,IN_FOURIER_SPACE, dt, L):
    """
    Python equivalent of the C CompCFLcondition.
    Keeps same names and structure.
    """
    if IN_FOURIER_SPACE[0] == 'y':
        u1 = np.fft.fftn(u1, norm="backward")
        u2 = np.fft.fftn(u2, norm="backward")
    #  umax on root
    umax = np.sqrt(np.max(u1**2 + u2**2))
    if IN_FOURIER_SPACE[0] == 'y':
        u1 = np.fft.fftn(u1, norm="backward")
        u2 = np.fft.fftn(u2, norm="backward")
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

    return CFL_break, umax, h, u1, u2


def test_909(w):

    w[0] += 200
    w[1] += 400
    return w

def InitFFTW(wo, w, tho, th,
             u1o, u1, u2o, u2):
    """
    Python equivalent of the C InitFFTW.
    Converts all given fields from physical space to Fourier space.
    Arrays are assumed to be 1D views of (N0, N1) data.
    """
    # Reshape to 2D (N0, N1), FFT, then flatten back
    wo[:]  = np.fft.fftn(wo, norm="backward")
    w[:]   = np.fft.fftn(w,  norm="backward")
    tho[:] = np.fft.fftn(tho, norm="backward")
    th[:]  = np.fft.fftn(th,  norm="backward")
    u1o[:] = np.fft.fftn(u1o, norm="backward")
    u1[:]  = np.fft.fftn(u1,  norm="backward")
    u2o[:] = np.fft.fftn(u2o, norm="backward")
    u2[:]  = np.fft.fftn(u2,  norm="backward")

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
wn = None
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
    global Filter_noiselevel, IN_FOURIER_SPACE, wn, wn_abs, t, iout
    global iter_print, iter_norms, CFL_break, Integral1, Integral2, Integral3
    global WuQ1, WuQ2, WuQ3, ThQ1, ThQ2, ThQ3, UQ1, UQ2, UQ3, DIV, umax
    global AveErrmaxtoinit, w, th

    cfg = read_input("input.ini")

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
    w    = np.zeros((N0, N1), dtype=np.complex128)
    wo   = np.zeros((N0, N1), dtype=np.complex128)

    k1_w_old = np.zeros((N0, N1), dtype=np.complex128)
    k1_th_old = np.zeros((N0, N1), dtype=np.complex128)
    k1_w = np.zeros((N0, N1), dtype=np.complex128)
    k1_th = np.zeros((N0, N1), dtype=np.complex128)
    k2_w = np.zeros((N0, N1), dtype=np.complex128)
    k2_th = np.zeros((N0, N1), dt`ype=np.complex128)

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
          InitVariables(w, th)

        w = Read_VorTemp()

            # scatter to ranks (equal blocks of length DIM)
        th, u1, u2 = MakeAverageZero(th, u1, u2)
        IN_FOURIER_SPACE[0] = 'n'
        w, u1, u2 = ComputeUVfromVort(w, u1, u2)
        u1, u2, th, w = output_data(u1, u2, th, w)

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
        w, u1, u2 = ComputeUVfromVort(w, u1, u2)
    else:
        print("This input for restart is invalid")
        return

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

        CFL_break, umax, h, u1, u2 = CompCFLcondition(
            u1, u2,
            IN_FOURIER_SPACE, dt, L
        )
        if CFL_break == 1:
           u1, u2, th, w = output_data(u1, u2, th, w )
           print("CFL violated; aborting.")
           return

        # time stepper
        if METHOD == 1:  # BE/IMEX
            w, th = do_IMEX(k1_w, k1_th, u1, u2, th, w, g1tmp, f3tmp, t, wn, N3,N4,tmp1, tmp2, tmp3, tmp4)
            w, u1, u2 = ComputeUVfromVort(w, u1, u2)

        elif METHOD == 2:  # BDF2
            if it == iter_start:

              k1_w_old, k1_th_old = RHS_k1_w_th(u1o, u2o, tho, wo, k1_w_old, k1_th_old, dt,t, g1tmp, f3tmp, N3,N4, tmp1, tmp2, tmp3, tmp4)
              w, th = do_RK2(u1, u2, w, th,
                           k1_w, k1_th, k2_w, k2_th,
                           w_tp, th_tp, u1_tp, u2_tp,
                           g1tmp, f3tmp,
                           t, dt, L, ConvTest,
                           wn,N3,N4, tmp1, tmp2, tmp3, tmp4)
              w, u1, u2 = ComputeUVfromVort(w, u1, u2)
            else:
              k1_w, k1_th, wn,thn = do_BDF2(u1, u2,wo, w, wn, tho, th, thn,k1_w_old, k1_th_old,k1_w, k1_th, tmp1, tmp2, tmp3, tmp4, g1tmp, f3tmp,N3,N4)

              wn, u1, u2 = ComputeUVfromVort(wn, u1, u2)

        elif METHOD == 3:  # RK2
            w, th = do_RK2(u1, u2, w, th,
                           k1_w, k1_th, k2_w, k2_th,
                           w_tp, th_tp, u1_tp, u2_tp,
                           g1tmp, f3tmp,
                           t, dt, L, ConvTest,
                           wn, N1, N3,N4, tmp1, tmp2, tmp3, tmp4)
            w, u1, u2 = ComputeUVfromVort(w, u1, u2)

        # advance time
        t += dt
        if USE_Filter == 1:
            u1 = filter_exp(u1, Filter_alpha)
            u2 = filter_exp(u2, Filter_alpha)
            th = filter_exp(th, Filter_alpha)
            w = filter_exp(w,  Filter_alpha)
        elif USE_Filter == 2:
            u1 = filter_Krasny(u1, Filter_noiselevel)
            u2 = filter_Krasny(u2, Filter_noiselevel)
            th = filter_Krasny(th, Filter_noiselevel)
            w = filter_Krasny(w,  Filter_noiselevel)

        # norms
        (WuQ1, WuQ2, WuQ3,ThQ1, ThQ2, ThQ3, UQ1, UQ2, UQ3, Integral1, Integral2, Integral3,DIV, th_H1, th_H2) = CompNormsInFourierSpace(u1, u2, th,wn, t, dt)
        AveErrmaxtoinit, th= ComputeThetaAverage(th)

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
            u1, u2, th, w = output_data(u1, u2, th, w)


if __name__ == "__main__":
    main()
