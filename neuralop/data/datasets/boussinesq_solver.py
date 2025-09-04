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
