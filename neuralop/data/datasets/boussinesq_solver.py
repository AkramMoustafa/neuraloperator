""" 
August 13, 2025 
 * Xiaoming Zheng & Akram Moustafa
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

def g1_func(t, x, y,ConvTest, a, nu):

    if(ConvTest):
        u1, u2, D1w, D2w, D11w, D1th, Dtw;
        # //w=  -cos(t)*2*a*cos(2*a*x)*pow(sin(a*y),2)  -cos(t)*pow(sin(a*x),2)*2*a*cos(2*a*y)
        # //theta= cos(t)*pow(sin(a*x),2)*sin(2*a*y)

        u1   = cos(t)*pow(sin(a*x),2)*sin(2*a*y)
        u2   =-cos(t)*sin(2*a*x)*pow(sin(a*y),2)
        Dtw  = sin(t)*2*a*cos(2*a*x)*pow(sin(a*y),2) +sin(t)*pow(sin(a*x),2)*2*a*cos(2*a*y)
        D1w  = cos(t)*4*a*a*sin(2*a*x)*pow(sin(a*y),2)-cos(t)*sin(2*a*x)*2*a*a*cos(2*a*y)
        D2w  =-cos(t)*2*a*a*cos(2*a*x)*sin(2*a*y) + cos(t)*pow(sin(a*x),2)*4*a*a*sin(2*a*y)
        D11w = cos(t)*8*a*a*a*cos(2*a*x)*pow(sin(a*y),2) -cos(t)*cos(2*a*x)*4*a*a*a*cos(2*a*y)
        D1th = cos(t)*a*sin(2*a*x)*sin(2*a*y)

        return Dtw + u1*D1w + u2*D2w - D1th - nu*D11w
    else:
        return 0
     
def f3_func( t,  x,  y, ConvTest, a):

    if( ConvTest):

        u1   = cos(t)*pow(sin(a*x),2)*sin(2*a*y)
        u2   =-cos(t)*sin(2*a*x)*pow(sin(a*y),2)
        th   = u1
        Dtth =-sin(t)*pow(sin(a*x),2)*sin(2*a*y)
        D1th = cos(t)*a*sin(2*a*x)*sin(2*a*y)
        D2th = cos(t)*pow(sin(a*x),2)*2*a*cos(2*a*y)
        D22th=-cos(t)*pow(sin(a*x),2)*4*a*a*sin(2*a*y)

        return Dtth + u1*D1th + u2*D2th + u2- eta*D22th
    else:
        return 0
def f2_func( t,  x,  y, a, ConvTest):
    if( ConvTest ):

        u1   = cos(t)*pow(sin(a*x),2)*sin(2*a*y)
        u2   =-cos(t)*sin(2*a*x)*pow(sin(a*y),2)
        th   = u1
        p   = cos(t)*sin(2*a*x)*sin(2*a*y)
        Dtu2 = sin(t)*sin(2*a*x)*pow(sin(a*y),2)
        D1u2 =-cos(t)*2*a*cos(2*a*x)*pow(sin(a*y),2)
        D2u2 =-cos(t)*sin(2*a*x)*a*sin(2*a*y)
        D11u2= cos(t)*4*a*a*sin(2*a*x)*pow(sin(a*y),2)
        D2p  = cos(t)*sin(2*a*x)*2*a*cos(2*a*y)

        return Dtu2 +u1*D1u2 + u2*D2u2 + D2p - nu*D11u2 - th
    else:
        return 0

def f1_func( t,  x,  y, a,ConvTest):

    if( ConvTest ):

        u1   = cos(t)*pow(sin(a*x),2)*sin(2*a*y)
        u2   =-cos(t)*sin(2*a*x)*pow(sin(a*y),2)
        p   = cos(t)*sin(2*a*x)*sin(2*a*y)
        Dtu1 =-sin(t)*pow(sin(a*x),2)*sin(2*a*y)
        D1u1 = cos(t)*a*sin(2*a*x)*sin(2*a*y)
        D2u1 = cos(t)*pow(sin(a*x),2)*2*a*cos(2*a*y)
        D11u1= cos(t)*2*a*a*cos(2*a*x)*sin(2*a*y)
        D1p  = cos(t)*2*a*cos(2*a*x)*sin(2*a*y)

        return Dtu1 +u1*D1u1 + u2*D2u1 + D1p - nu*D11u1
    else:
        return 0
def winitFunc( t,  x,  y, a, ConvTest):

  if(ConvTest):
     return -cos(t)*2*a*cos(2*a*x)*pow(sin(a*y),2)-cos(t)*pow(sin(a*x),2)*2*a*cos(2*a*y)
  else:
     return -WuEpsi*cos(t)*2*a*cos(2*a*x)*pow(sin(a*y),2)-WuEpsi*cos(t)*pow(sin(a*x),2)*2*a*cos(2*a*y);


def prinitFunc( t,  x,  y):
  if(ConvTest):
     return cos(t)*sin(2*a*x)*sin(2*a*y);
  else:
     return WuEpsi*cos(t)*pow(sin(a*x),2)*sin(2*a*y);

def thinitFunc( t,  x,  y):
# // initial value of theta
  if(ConvTest):
     return cos(t)*pow(sin(a*x),2)*sin(2*a*y) 
  else:
     nX=4
     tmp0 = 4*(1.e0 + 1.e0/4 + 1.e0/9 + 1.e0/16)
     tmp1 = tmp0
     tmp2 = tmp0
     for i in range(nX):
         tmp1 -= 4.e0/(i*i)*cos(i*x)
         tmp2 -= 4.e0/(i*i)*cos(i*y)
    #  tmp1 is the partial sum of 2pi^2/3- x(2pi-x) on [0,2pi]
     return WuEpsi*cos(t)* (tmp1 * tmp2 - tmp0*tmp0)
     

def u1initFunc( t,  x,  y):
  if(ConvTest):
     return cos(t)*pow(sin(a*x),2)*sin(2*a*y)
  else:
     return WuEpsi*cos(t)*pow(sin(a*x),2)*sin(2*a*y)


def u2initFunc( t,  x,  y):
  if(ConvTest):
     return -cos(t)*sin(2*a*x)*pow(sin(a*y),2)
  else:
     return -WuEpsi*cos(t)*sin(2*a*x)*pow(sin(a*y),2)


def AfterNonlinear( in_u1, in_u2, in_w, in_th, time_spec):

    in_u1 = np.fft.fftn(in_u1, norm=None).ravel()
    in_u2 = np.fft.fftn(in_u2, norm=None).ravel()
    in_th = np.fft.fftn(in_th, norm=None).ravel()
    in_w = np.fft.fftn(in_w, norm=None).ravel()



def PreNonlinear( in_u1, in_u2, in_w, in_th, 
                  N4, N3,  time_spec):

    # Above all, convert input quantities to Physical space:
    in_u1 = np.fft.ifftn(in_u1.reshape(local_n0, N1), norm=None).real.ravel()
    in_u2 = np.fft.ifftn(in_u2.reshape(local_n0, N1), norm=None).real.ravel()
    in_th = np.fft.ifftn(in_th.reshape(local_n0, N1), norm=None).real.ravel()
    in_w  = np.fft.ifftn(in_w.reshape(local_n0, N1), norm=None).real.ravel()
    # divide them by DIM because FFTW does not perform it:
    fftw_normalize(in_u1) 
    fftw_normalize(in_u2)
    fftw_normalize(in_th) 
    fftw_normalize(in_w)
    # initialize N5 and N6:
    for i in range(local_n0):
        for j in range(N1):
            jj = i * N1 + j
            N4[jj] = 0.0 + 0.0j
            N3[jj] = 0.0 + 0.0j
    

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
            #    in_u1[jj][0] = u1initFunc(time_spec, x,y);
            #    in_u1[jj][1] = 0;
            #    in_u2[jj][0] = u2initFunc(time_spec, x,y);
            #    in_u2[jj][1] = 0;
            #    in_th[jj][0] = thinitFunc(time_spec, x,y);
            #    in_th[jj][1] = 0;
            #    set done
              
    
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
                        N3 += z
                        

                        # N4 = z = hat( D_1(u1*vort) + D_2(u2*vort) )
                        z = (-k1*tmp3[jj].imag - k2*tmp4[jj].imag) +  (k1*tmp3[jj].real + k2*tmp4[jj].real)
                        N4= z


def Read_VorTemp(N0=128, N1 = 128):
    w_all = np.empty(N0 * N1, dtype=np.complex128)
    import numpy as np
    with open("initial_vorticity.txt", "r") as f: 
        for j in range(N1): 
            line = f.readline().split()
            for i in range(N0):
                jj = i*N1 + j
                w_all[jj] = complex(float(line[i]), 0.0)

    th_all = np.empty(N0 * N1, dtype=np.complex128)
    import numpy as np
    with open("initial_temp.txt", "r") as f: 
        for j in range(N1): 
            line = f.readline().split()
            for i in range(N0):
                jj = i*N1 + j
                th_all[jj] = complex(float(line[i]), 0.0)


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
def RHS_k1_w_th( u1_in,  u2_in, th_in,  w_in,k1_w,   k1_th,time_spec):
    """
    Purpose: find k1_w =nabla dot(u w) + nabla_x theta + g1
    g1 = nnablatimes (f1,f2)
    k1_th=-nabla\dot(u theta) -u2 + f3
    work in Fourier space
    """
    g1tmp_local  = np.empty((local_n0, N1), dtype=np.complex128)
    f3tmp_local  = np.empty((local_n0, N1), dtype=np.complex128)
 
    # g1tmp is external term of equation of w_t
    # f3tmp is external term of equation of theta_t
    for i in range(local_no):
       for i in range(N1):
          jj = i*N1+j
          x=L*(local_start+i)/N0
          y=L*j/N1

          g1tmp_local[jj] = g1_func(time_spec, x, y) + 0.0j
          f3tmp_local[jj] = f3_func(time_spec, x, y) + 0.0j

    g1tmp_local = np.fft.fftn(g1tmp_local.reshape(local_n0, N1), norm=None).ravel()
    f3tmp_local = np.fft.fftn(f3tmp_local.reshape(local_n0, N1), norm=None).ravel()

    RHS_BE(k1_w, k1_th, u1_in, u2_in, th_in, w_in, g1tmp_local, f3tmp_local, time_spec)

def do_RK2(u1_in,  u2_in, w_in,  th_in):

    # // RK2 method 
    # // has 2 unknowns to solve: w (vorticity) and th (theta)
    # // work in Fourier space
    k1_w  = np.empty((local_n0, N1), dtype=np.complex128)
    k1_th = np.empty((local_n0, N1), dtype=np.complex128)
    k2_w  = np.empty((local_n0, N1), dtype=np.complex128)
    k2_th = np.empty((local_n0, N1), dtype=np.complex128)
    # //intermediate values:
    w_tp  = np.empty((local_n0, N1), dtype=np.complex128)
    th_tp = np.empty((local_n0, N1), dtype=np.complex128)
    u1_tp = np.empty((local_n0, N1), dtype=np.complex128)
    u2_tp = np.empty((local_n0, N1), dtype=np.complex128)
    # // RHS of 3 equations:
    g1tmp_local  = np.empty((local_n0, N1), dtype=np.complex128)
    f3tmp_local  = np.empty((local_n0, N1), dtype=np.complex128)
    jj = 0

    # //step 1
    # //g1tmp is external term of equation of w_t
    # //f3tmp is external term of equation of theta_t
    for i in range(local_n0):
        
       for j in range(N1):
          jj = i*N1+j
          x=L*(local_start+i)/N0
          y=L*j/N1
          g1tmp_local[jj] = g1_func(t, x, y) + 0.0j
          f3tmp_local[jj] = f3_func(t, x, y) + 0.0j

    g1tmp_local = np.fft.fftn(g1tmp_local.reshape(local_n0, N1), norm=None).ravel()
    f3tmp_local = np.fft.fftn(f3tmp_local.reshape(local_n0, N1), norm=None).ravel()

    RHS(k1_w, k1_th, u1_in, u2_in, th_in, w_in, g1tmp_local, f3tmp_local, t)
    in1_plus_a_in2(w_tp,  w_in,  k1_w,  dt)
    in1_plus_a_in2(th_tp, th_in, k1_th, dt)

    ComputeUVfromVort(w_tp,  u1_tp,  u2_tp)

    # step 2
    for i in range(local_n0):
       for j in range(N1):
          jj = i*N1+j
          x=L*(local_start+i)/N0
          y=L*j/N1
          g1tmp_local[jj] = g1_func(t+dt, x, y) + 0.0j
          f3tmp_local[jj] = f3_func(t+dt, x, y) + 0.0j


    g1tmp_local = np.fft.fftn(g1tmp_local.reshape(local_n0, N1), norm=None).ravel()
    f3tmp_local = np.fft.fftn(f3tmp_local.reshape(local_n0, N1), norm=None).ravel()
    
    # k2 is F(tn+dt, u_n+dt*k1) of ODE u_t=F(t,u)
    RHS(k2_w, k2_th, u1_tp, u2_tp, th_tp, w_tp, g1tmp_local, f3tmp_local, t+dt)

    for i in range(local_n0):
        for j in range(N1):
            jj = i*N1+j;

            w_in[jj]  += dt * (k1_w[jj]  + k2_w[jj] ) / 2.0
            th_in[jj] += dt * (k1_th[jj] + k2_th[jj] ) / 2.0



def RHS_k1_w_th(u1_in, u2_in, th_in, w_in, k1_w,  k1_th, time_spec):
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
            g1tmp_local[jj] = g1_func(t+dt, x, y) + 0.0j
            f3tmp_local[jj] = f3_func(t+dt, x, y) + 0.0j

    g1tmp_local = np.fft.fftn(g1tmp_local.reshape(local_n0, N1), norm=None).ravel()
    f3tmp_local = np.fft.fftn(f3tmp_local.reshape(local_n0, N1), norm=None).ravel()

    RHS_BE(k1_w, k1_th, u1_in, u2_in, th_in, w_in, g1tmp_local, f3tmp_local, time_spec);

    fftw_free(g1tmp_local); fftw_free(f3tmp_local)



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

            w_in[jj][0]  = ( w_in[jj][0]  + dt*k1_w[jj][0] )/tmp1
            w_in[jj][1]  = ( w_in[jj][1]  + dt*k1_w[jj][1] )/tmp1
            th_in[jj][0] = ( th_in[jj][0] + dt*k1_th[jj][0] )/tmp2
            th_in[jj][1] = ( th_in[jj][1] + dt*k1_th[jj][1] )/tmp2
    

def ComputeUVfromVort(w_in, u1_in, u2_in):
    # work in Fourier space
    # -Laplace psi = w, that is, \hat\psi = \hat{w}/|k|^2
    # u1 = D_y \psi,  or \hat{u1} = i*k2*\hat{\psi}  = i*k2/|k|^2 \hat{w}
    # u2 =-D_x \psi,  or \hat{u1} = -i*k1*\hat{\ps} =-i*k1/|k|^2 \hat{w}

    if( IN_FOURIER_SPACE[0]=='n'):
    #    DFT to convert data to Fourier space:
       w_in = np.fft.fftn(w_in.reshape(local_n0, N1), norm=None).ravel()	
    

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


def do_BDF2(u1_in,u2_in, wo_in,   w_in, wn_in, tho_in,  th_in, thn_in, k1_w_old, k1_th_old):

    # BDF2 method on linear terms, 2nd order extrapolation on nonlinear terms
    # (3*u1n_in - 4*u1_in + 1*u1o_in)/(2dt) = f
    # Add Shen and Yang 2010 's stability term
    # has 3 unknowns to solve: u1, u2, th (theta), pr (pressure)
    # work in Fourier space
    # k1_pr_old is not used
    k1_w        = np.empty(alloc_local, dtype=np.complex128)
    k1_th       = np.empty(alloc_local, dtype=np.complex128) 
    # RHS of 3 equations:
    g1tmp_local = np.empty(alloc_local, dtype=np.complex128)
    f3tmp_local = np.empty(alloc_local, dtype=np.complex128)

    # g1tmp is external term of equation of w_t
    # f3tmp is external term of equation of theta_t
    for i in range(local_n0):
       for j in range(N1):
          jj = i*N1+j
          x=L*(local_start+i)/N0
          y=L*j/N1
          g1tmp_local[jj] = g1_func(t, x, y) + 0.0j
          f3tmp_local[jj] = f3_func(t, x, y) + 0.0j

    g1tmp_local = np.fft.rfftn(g1tmp_local.reshape(local_n0, N1), norm=None).ravel()
    f3tmp_local = np.fft.rfftn(f3tmp_local.reshape(local_n0, N1), norm=None).ravel()
    # BDF2 should call RHS_BE
    RHS_BE(k1_w, k1_th, u1_in, u2_in, th_in, w_in, g1tmp_local, f3tmp_local, t)

    for i in range(local_n0): 
        for j in range(N1):
            jj = i*N1+j
            k1 = wn[local_start + i]
            k2 = wn[j]
            k1sq =pow(k1,2)
            k2sq =pow(k2,2)
            ksq = k1sq + k2sq

	#  below is the new u1, u2, theta:
            tmp1 = 1.5/dt + nu*k1sq
            tmp2 = 1.5/dt + eta*k2sq
            wn_in[jj]  = (2 * k1_w[jj]  - k1_w_old[jj]  + (2 * w_in[jj]  - 0.5 * wo_in[jj]) / dt) / tmp1
            thn_in[jj] = (2 * k1_th[jj] - k1_th_old[jj] + (2 * th_in[jj] - 0.5 * tho_in[jj]) / dt) / tmp2

    k1_w_old[:] = k1_w
    k1_th_old[:] = k1_th
    wo_in[:]  = w_in
    w_in[:]   = wn_in
    tho_in[:] = th_in
    th_in[:]  = thn_in


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

        MPI_Gather(u1_local, alloc_local, MPI_DOUBLE_COMPLEX, u1, alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD)
        MPI_Gather(u2_local, alloc_local, MPI_DOUBLE_COMPLEX,u2, alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD)
        MPI_Gather(th_local, alloc_local, MPI_DOUBLE_COMPLEX,th, alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD)


      #   Note: FFTW does not divide by N0*N1 after fotward DFT, so we have to divide it
        DIV= 0;
        if( my_rank == 0):
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
                  if( fabs(k1) >0.1):
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
   

                  tmp3 = k1sq* ( pow(th[jj][0],2)) 
                  D1th_L2sq += tmp3

                  DIV += abs(k1 * u1[jj] + k2 * u2[jj])**2

            # two scalars used in Parseval-Plancherel identity in integrals:
            tmp4 = 2*PI/DIM
            tmp5 = pow(tmp4,2)

            tilde_u_H1sq  *= tmp5
            tilde_th_H1sq *= tmp5
            D2_tilde_th_H2sq *= tmp5

            WuQ2 = sqrt(tilde_u_H1sq) + sqrt(tilde_th_H1sq) # First quantity in Theorem 2
            WuQ3 = t*( tilde_u_H1sq + tilde_th_H1sq ) # Second quantity in Theorem 2

            u_H2sq    *=tmp5
            th_H2sq   *=tmp5
            D1u_H2sq  *=tmp5
            D2th_H2sq *=tmp5
            D1th_L2sq *=tmp
            th_H2 = sqrt(th_H2sq)		#H2 norm of theta
            th_H1 = tmp4*sqrt(th_H1)	#H1 norm of theta
            ThQ1 = tmp4*sqrt(th_L2)	#L2 norm of theta
            ThQ2 = tmp4*sqrt(tilde_th_L2)	#L2 norm of (theta - \bar{theta})
            ThQ3 = sqrt(tilde_th_H1sq)	#H1 norm of theta-bar{theta}
            UQ1 = tmp4*sqrt(u_L2)		#L2 norm of u
            UQ2 = tmp4*sqrt(tilde_u_L2)	#L2 norm of (u - \bar{u})
            UQ3  = sqrt(tilde_u_H1sq)	#H1 norm of u-bar{u}

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


            DIV = sqrt(DIV)*tmp4


def read_input(filename="input"):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f]

    METHOD           = int(lines[1])
    TMAX             = float(lines[2])
    dt               = float(lines[3])
    dt_print         = float(lines[4])
    dt_norms         = float(lines[5])
    eta              = float(lines[6])
    nu               = float(lines[7])
    WuEpsi           = float(lines[8])
    N0               = int(lines[9])
    N1               = int(lines[10])
    restart          = lines[11]
    irestart         = int(lines[12])
    ConvTest         = int(lines[13])
    ShenYang         = float(lines[14])
    USE_Filter       = int(lines[15])
    Filter_alpha     = float(lines[16])
    Filter_noiselevel= float(lines[17])
    DIM              = N0 * N1

    return locals() 
def read_input(filename="input"):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f]

    METHOD           = int(lines[1])
    TMAX             = float(lines[2])
    dt               = float(lines[3])
    dt_print         = float(lines[4])
    dt_norms         = float(lines[5])
    eta              = float(lines[6])
    nu               = float(lines[7])
    WuEpsi           = float(lines[8])
    N0               = int(lines[9])
    N1               = int(lines[10])
    restart          = lines[11]
    irestart         = int(lines[12])
    ConvTest         = int(lines[13])
    ShenYang         = float(lines[14])
    USE_Filter       = int(lines[15])
    Filter_alpha     = float(lines[16])
    Filter_noiselevel= float(lines[17])
    DIM              = N0 * N1

    return locals() 
    
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
        #   convert to physical space:
          fftw_mpi_execute_dft(planC2R, th_local, th_local);
          fftw_normalize(th_local)
          th_local = np.fft.irfftn(th_local.reshape(local_n0, N1), s=(local_n0, N1), norm="forward").ravel()
      

       MPI_Gather(th_local, alloc_local, MPI_DOUBLE_COMPLEX,th_all, alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD)

       if( my_rank == 0 ):
        #    compute average:
           ave = 0;
           for j in range(N1) :
               for i in range(N0):
                   jj=i*N1+j
                   ave += th_all[jj]

           print("average = %12.5e\n", ave)
           ave = ave/N0/N1
           for j in range(N1):
               for i in range(N0):
                   jj=i*N1+j
                   th_all[jj].real -= ave

       MPI_Scatter(th_all,alloc_local, MPI_DOUBLE_COMPLEX, th_local, alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD)

       if( IN_FOURIER_SPACE[0]=='y'):
          # convert to Fourier space:
        
            th_local = np.fft.rfftn(
                th_local.reshape(local_n0, N1),
                s=(local_n0, N1),
                norm=None
            ).ravel()

def ComputeThetaAverage(th_local, th_all):


       if( IN_FOURIER_SPACE[0]=='y'):

         
        th_local = np.fft.fftn(th_local.reshape(local_n0, N1),s=(local_n0, N1),norm=None).ravel()
        fftw_normalize(th_local)

       MPI_Gather(th_local, alloc_local, MPI_DOUBLE_COMPLEX,th_all,   alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD)

       if( my_rank == 0 ):
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

              tmp = fabs(tmp/N0- InitAverage[j])
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

        fftw_normalize(u1_local)
        fftw_normalize(u2_local)
        fftw_normalize(th_local)
        fftw_normalize(w_local)
    

        #  Output data in physical space:
        MPI_Gather(u1_local, alloc_local, MPI_DOUBLE_COMPLEX,  u1_all,   alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
        MPI_Gather(u2_local, alloc_local, MPI_DOUBLE_COMPLEX,    u2_all,   alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
        MPI_Gather(th_local, alloc_local, MPI_DOUBLE_COMPLEX,  th_all,   alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
        MPI_Gather(w_local, alloc_local,  MPI_DOUBLE_COMPLEX,  w_all,   alloc_local,  MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
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


def CompCFLcondition(u1_local, u2_local, u1_all,   u2_all):

      if( IN_FOURIER_SPACE[0]=='y'):
        #   convert to physical space:
          fftw_mpi_execute_dft(planC2R, u1_local, u1_local);	
          fftw_mpi_execute_dft(planC2R, u2_local, u2_local);	
          fftw_normalize(u1_local);
          fftw_normalize(u2_local);
       

  #    gather data in physical space:
      MPI_Gather(u1_local, alloc_local, MPI_DOUBLE_COMPLEX,u1_all,   alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD)
      MPI_Gather(u2_local, alloc_local, MPI_DOUBLE_COMPLEX,u2_all,   alloc_local, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD)

      if( IN_FOURIER_SPACE[0]=='y'):
      #   convert to Fourier space:
        fftw_mpi_execute_dft(planR2C, u1_local, u1_local)
        fftw_mpi_execute_dft(planR2C, u2_local, u2_local)
       

    # The umax is only computed in my_rank=0 processor
      umax=0.e0
      for j in range(N1):
          for i in rnage(N0):
              jj=i*N1+j;
              tmp = pow(u1_all[jj],2)  + pow(u2_all[jj],2) 
              umax = max( umax, tmp )

      umax = sqrt(umax)
      h = L/N0
      tmp = dt*umax/h
      if tmp > 0.5:
          CFL_break = 1  # CFL condition breaks
          if rank == 0:
              print("***********************************")
              print(f"    t         = {t:12.5e}")
              print(f"    dt        = {dt:12.5e}")
              print(f"    umax      = {umax:12.5e}")
              print(f"    h         = {h:12.5e}")
              print(f"CFL=dt*umax/h = {tmp:12.5e}")
              print("CFL is too big, need to decrease dt")
              print("***********************************")
      else:
          CFL_break = 0

      # Broadcast CFL_break from rank 0 to all processes
      CFL_break = comm.bcast(CFL_break, root=0)



def InitVariables(w_local,  th_local):



    for i in range(local_n0):
        for  j in range(N1):
            jj=i*N1+j
            x=L*(local_start+i)/N0
            y=L*j/N1

            w_local[jj].real = winitFunc(0, x,y)
            w_local[jj].imag = 0

            th_local[jj].real= thinitFunc(0, x,y)
            th_local[jj].imag= 0


def InitFFTW( wo_local,   w_local,  tho_local,  th_local, u1o_local,  u1_local, u2o_local,  u2_local):


    wo_local = np.fft.rfftn(wo_local.reshape(local_n0, N1), s=(local_n0, N1), norm=None).ravel()
    w_local  = np.fft.rfftn(w_local.reshape(local_n0, N1),  s=(local_n0, N1), norm=None).ravel()
    tho_local= np.fft.rfftn(tho_local.reshape(local_n0, N1),s=(local_n0, N1), norm=None).ravel()
    th_local = np.fft.rfftn(th_local.reshape(local_n0, N1), s=(local_n0, N1), norm=None).ravel()
    u1o_local= np.fft.rfftn(u1o_local.reshape(local_n0, N1),s=(local_n0, N1), norm=None).ravel()
    u1_local = np.fft.rfftn(u1_local.reshape(local_n0, N1), s=(local_n0, N1), norm=None).ravel()
    u2o_local= np.fft.rfftn(u2o_local.reshape(local_n0, N1),s=(local_n0, N1), norm=None).ravel()
    u2_local = np.fft.rfftn(u2_local.reshape(local_n0, N1), s=(local_n0, N1), norm=None).ravel()

def test_909(w):


    w[0] += 200
    w[1] += 400

