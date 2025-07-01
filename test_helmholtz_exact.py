import numpy as np
from numpy.testing import assert_allclose
from helmholtz_exact import DielectricPlaneWaveCoefficients, far_field_from_plane_wave, U_tot_from_coeffs
from netgen.geom2d import SplineGeometry
from ngsolve import Mesh, H1, BilinearForm, SymbolicBFI, grad, pml 
from ngsolve import GridFunction, LinearForm, SymbolicLFI, specialcf, Integrate, BND
import ngsolve as ns



NUMBER_OF_MODES = 10
eps_r = 4.   
k = 200. 
x_sc = 0.
y_sc = 30E-3
R_sc = 15E-3
xy_c = np.array([ x_sc, y_sc])
U = 1. + 0.j
theta_inc = np.pi + np.pi/4
N_R = 72
THETA_R = np.linspace(0, 2*np.pi, N_R, endpoint=False)
x_hat = np.column_stack([np.cos(THETA_R), np.sin(THETA_R)])

N_E = 36
THETA_E = np.linspace(0, 2*np.pi, N_E, endpoint=False)
D = np.column_stack([np.cos(THETA_E), np.sin(THETA_E)])


def test_no_contrast():
    A, _ = DielectricPlaneWaveCoefficients(k, 1., R_sc, xy_c, U, np.pi/4, NUMBER_OF_MODES)
    assert_allclose(A, np.zeros(2*NUMBER_OF_MODES+1, dtype=np.complex128))


def test_total_field():
    geo = SplineGeometry()
    
    landa = 2*np.pi/k
    R_PML = np.sqrt(x_sc**2 + y_sc**2) + R_sc + 2*landa
    delta_PML = 2*landa
    
    background_ID = 1
    PML_ID = 2
    scatterer_ID = 3


    geo.AddCircle( (0,0), R_PML+delta_PML, leftdomain=PML_ID, bc="outerbnd") # computational domain with PML
    geo.AddCircle( (0,0), R_PML, leftdomain=background_ID, rightdomain=PML_ID) # computational domain

    # adding the scatterer:

    geo.AddCircle( (x_sc, y_sc), R_sc, leftdomain=scatterer_ID, rightdomain=background_ID, bc="scatterer")
 
    geo.SetMaterial(background_ID,"background")
    geo.SetMaterial(PML_ID,"PML")
    geo.SetMaterial(scatterer_ID,"scatterer")

    h_max = landa/8 #should also be k dependent

    geo.SetDomainMaxH(PML_ID,h_max)
    geo.SetDomainMaxH(background_ID,h_max)
    geo.SetDomainMaxH(scatterer_ID,h_max/eps_r )
    Omega = Mesh(geo.GenerateMesh())
    
    Omega.Curve(3)
    alpha = 0.3
    # the alpha probably will need to be adjusted
    Omega.SetPML(pml.Radial(rad=R_PML,alpha=alpha*1j,origin=(0,0)),"PML")

    # variational formulation
    polynomial_order = 4
    n_values = { "scatterer" : eps_r, "background" : 1, "PML" : 1}
    n = Omega.MaterialCF(n_values) 

    
    N = specialcf.normal(Omega.dim)


    H = H1(Omega, order = polynomial_order, complex=True)
    u = H.TrialFunction()
    v = H.TestFunction()

    a = BilinearForm(H)
    a += SymbolicBFI(grad(u)*grad(v) - n*k**2*u*v)
    a += SymbolicBFI(-1j*k*u*v,definedon=Omega.Boundaries("outerbnd"))


    a.Assemble()
    A_inv = a.mat.Inverse()

    theta_inc = THETA_E[20]
    d = np.array([np.cos(theta_inc), np.sin(theta_inc)])
    u_inc_ns = U*ns.exp( 1j*k*( d[0]*ns.x + d[1]*ns.y) )  #plane wave with U amplitude at origin
    l = LinearForm(H)
    l += SymbolicLFI((n-1)*k**2*u_inc_ns * v)
    l.Assemble()
    u_FEM = GridFunction(H)
    u_FEM.vec.data = A_inv * l.vec
    u_TOT = u_FEM + u_inc_ns
    

    L =  R_PML/np.sqrt(2)
    Nx = 100
    x = np.linspace(-L,L,Nx)
    y = np.linspace(-L,L,Nx)
    X, Y = np.meshgrid(x,y)
    A, B = DielectricPlaneWaveCoefficients(k, eps_r, R_sc, xy_c, 1., theta_inc, NUMBER_OF_MODES)
    Z = U_tot_from_coeffs(X=X, Y=Y, k=k, N=eps_r,c=xy_c, R=R_sc, U=1+0j,theta_inc=theta_inc, A=A, B=B)
    Z_FEM = u_TOT(Omega(X.flatten(),Y.flatten())).reshape(X.shape)

    assert_allclose(Z_FEM, Z, rtol=1E-2)

def test_single_farfield():
    geo = SplineGeometry()
    
    landa = 2*np.pi/k
    R_PML = np.sqrt(x_sc**2 + y_sc**2) + R_sc + 2*landa
    delta_PML = 2*landa
    R_int = ( np.sqrt(x_sc**2 + y_sc**2) + R_sc  + R_PML)/2
    background_ID = 1
    PML_ID = 2
    scatterer_ID = 3


    geo.AddCircle( (0,0), R_PML+delta_PML, leftdomain=PML_ID, bc="outerbnd") # computational domain with PML
    geo.AddCircle( (0,0), R_PML, leftdomain=background_ID, rightdomain=PML_ID) # computational domain

    # adding the scatterer:

    geo.AddCircle( (x_sc, y_sc), R_sc, leftdomain=scatterer_ID, rightdomain=background_ID, bc="scatterer")
    geo.SetMaterial(background_ID,"background")
    geo.SetMaterial(PML_ID,"PML")
    geo.SetMaterial(scatterer_ID,"scatterer")

    h_max = landa/8 #should also be k dependent

    geo.SetDomainMaxH(PML_ID,h_max)
    geo.SetDomainMaxH(background_ID,h_max)
    geo.SetDomainMaxH(scatterer_ID,h_max/eps_r )
    Omega = Mesh(geo.GenerateMesh())
    
    Omega.Curve(3)
    alpha = 0.3
    # the alpha probably will need to be adjusted
    Omega.SetPML(pml.Radial(rad=R_PML,alpha=alpha*1j,origin=(0,0)),"PML")

    # variational formulation
    polynomial_order = 4
    n_values = { "scatterer" : eps_r, "background" : 1, "PML" : 1}
    n = Omega.MaterialCF(n_values) 

    
    N = specialcf.normal(Omega.dim)


    H = H1(Omega, order = polynomial_order, complex=True)
    u = H.TrialFunction()
    v = H.TestFunction()

    a = BilinearForm(H)
    a += SymbolicBFI(grad(u)*grad(v) - n*k**2*u*v)
    a += SymbolicBFI(-1j*k*u*v,definedon=Omega.Boundaries("outerbnd"))


    a.Assemble()
    A_inv = a.mat.Inverse()
    theta_inc = THETA_E[20]
    d = np.array([np.cos(theta_inc), np.sin(theta_inc)])
    u_inc_ns = U*ns.exp( 1j*k*( d[0]*ns.x + d[1]*ns.y) )  #plane wave with U amplitude at origin
    l = LinearForm(H)
    l += SymbolicLFI((n-1)*k**2*u_inc_ns * v)
    l.Assemble()
    u_FEM = GridFunction(H)
    u_FEM.vec.data = A_inv * l.vec

    # far_field computation
    ff_u = np.zeros(N_R, dtype=np.complex128)
    ff_du = np.zeros(N_R, dtype=np.complex128)
    for i in range(N_R):
        d_r = x_hat[i,:]
        E = ns.exp(-1j* k *( ns.x*d_r[0]+ ns.y*d_r[1]))
        dphi = -1j*k*( d_r[0]*N[0] + d_r[1]*N[1] ) * E
        H_out = H1(Omega, order=polynomial_order, complex=True, definedon=Omega.Materials("background"))
        trace_E = GridFunction(H_out)
        trace_E.vec[:] = 0
        trace_E.Set(E, BND, definedon=Omega.Boundaries("scatterer"))
        ff_u[i] = Integrate( dphi * u_FEM, Omega, order= polynomial_order+1, definedon=Omega.Boundaries("scatterer"))
        ff_du[i] = Integrate(grad(trace_E)*grad(u_FEM)-k**2*trace_E*u_FEM, Omega, order= polynomial_order+1, definedon= Omega.Materials("background"))
    FF_FEM = np.exp(1j*np.pi/4)/np.sqrt(8*np.pi*k)*(ff_u + ff_du)

    

    L =  R_PML/np.sqrt(2)
    Nx = 100
    FF = far_field_from_plane_wave(theta_E=np.array([theta_inc]), theta_R=THETA_R, k=k, R=R_sc, c=xy_c, N=eps_r, M=NUMBER_OF_MODES)
    FF = FF[:,0]

    assert_allclose(FF_FEM, FF, rtol=1E-3)

def test_full_far_field_plane_wave():
    geo = SplineGeometry()
    
    landa = 2*np.pi/k
    R_PML = np.sqrt(x_sc**2 + y_sc**2) + R_sc + 2*landa
    delta_PML = 2*landa
    R_int = ( np.sqrt(x_sc**2 + y_sc**2) + R_sc  + R_PML)/2
    background_ID = 1
    PML_ID = 2
    scatterer_ID = 3


    geo.AddCircle( (0,0), R_PML+delta_PML, leftdomain=PML_ID, bc="outerbnd") # computational domain with PML
    geo.AddCircle( (0,0), R_PML, leftdomain=background_ID, rightdomain=PML_ID) # computational domain

    # adding the scatterer:

    geo.AddCircle( (x_sc, y_sc), R_sc, leftdomain=scatterer_ID, rightdomain=background_ID, bc="scatterer")
    geo.SetMaterial(background_ID,"background")
    geo.SetMaterial(PML_ID,"PML")
    geo.SetMaterial(scatterer_ID,"scatterer")

    h_max = landa/8 #should also be k dependent

    geo.SetDomainMaxH(PML_ID,h_max)
    geo.SetDomainMaxH(background_ID,h_max)
    geo.SetDomainMaxH(scatterer_ID,h_max/eps_r )
    Omega = Mesh(geo.GenerateMesh())
    
    Omega.Curve(3)
    alpha = 0.3
    # the alpha probably will need to be adjusted
    Omega.SetPML(pml.Radial(rad=R_PML,alpha=alpha*1j,origin=(0,0)),"PML")

    # variational formulation
    polynomial_order = 4
    n_values = { "scatterer" : eps_r, "background" : 1, "PML" : 1}
    n = Omega.MaterialCF(n_values) 

    
    N = specialcf.normal(Omega.dim)


    H = H1(Omega, order = polynomial_order, complex=True)
    u = H.TrialFunction()
    v = H.TestFunction()

    a = BilinearForm(H)
    a += SymbolicBFI(grad(u)*grad(v) - n*k**2*u*v)
    a += SymbolicBFI(-1j*k*u*v,definedon=Omega.Boundaries("outerbnd"))


    a.Assemble()
    A_inv = a.mat.Inverse()
    FF_FEM = np.zeros((N_R, N_E), dtype=np.complex128)
    for j in range(N_E):
        d = D[j,:]
        u_inc_ns = U*ns.exp( 1j*k*( d[0]*ns.x + d[1]*ns.y) )  #plane wave with U amplitude at origin
        l = LinearForm(H)
        l += SymbolicLFI((n-1)*k**2*u_inc_ns * v)
        l.Assemble()
        u_FEM = GridFunction(H)
        u_FEM.vec.data = A_inv * l.vec


        # far_field computation
        ff_u = np.zeros(N_R, dtype=np.complex128)
        ff_du = np.zeros(N_R, dtype=np.complex128)
        for i in range(N_R):
            d_r = x_hat[i,:]
            E = ns.exp(-1j* k *( ns.x*d_r[0]+ ns.y*d_r[1]))
            dphi = -1j*k*( d_r[0]*N[0] + d_r[1]*N[1] ) * E
            H_out = H1(Omega, order=polynomial_order, complex=True, definedon=Omega.Materials("background"))
            trace_E = GridFunction(H_out)
            trace_E.vec[:] = 0
            trace_E.Set(E, BND, definedon=Omega.Boundaries("scatterer"))
            ff_u[i] = Integrate( dphi * u_FEM, Omega, order= polynomial_order+1, definedon=Omega.Boundaries("scatterer"))
            ff_du[i] = Integrate(grad(trace_E)*grad(u_FEM)-k**2*trace_E*u_FEM, Omega, order= polynomial_order+1, definedon= Omega.Materials("background"))
        FF_FEM[:,j] = np.exp(1j*np.pi/4)/np.sqrt(8*np.pi*k)*(ff_u + ff_du)

    

    L =  R_PML/np.sqrt(2)
    Nx = 100
    FF = far_field_from_plane_wave(theta_E=THETA_E, theta_R=THETA_R, k=k, R=R_sc, c=xy_c, N=eps_r, M=NUMBER_OF_MODES)
    assert_allclose(FF_FEM,FF,rtol=1E-3)
