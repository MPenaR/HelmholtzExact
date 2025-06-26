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
theta_inc = 0.

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

    theta_inc = np.pi + np.pi/4
    d = np.array([np.cos(theta_inc), np.sin(theta_inc)])
    u_inc_ns = U*ns.exp( 1j*k*( d[0]*ns.x + d[1]*ns.y) )  #plane wave with U amplitude at origin
    l = LinearForm(H)
    l += SymbolicLFI((n-1)*k**2*u_inc_ns * v)
    l.Assemble()
    u_FEM = GridFunction(H)
    u_FEM.vec.data = A_inv * l.vec
    

    # ploting the FEM solution:
    vertices = np.array([ V.point for V in Omega.vertices])
    x_points = vertices[:,0]
    y_points = vertices[:,1]
    u_TOT = u_FEM + u_inc_ns
    z = u_TOT(Omega(x_points, y_points))
    L =  R_PML/np.sqrt(2)
    Nx = 100
    x = np.linspace(-L,L,Nx)
    y = np.linspace(-L,L,Nx)
    X, Y = np.meshgrid(x,y)
    A, B = DielectricPlaneWaveCoefficients(k, eps_r, R_sc, xy_c, 1., theta_inc, NUMBER_OF_MODES)
    Z = U_tot_from_coeffs(X=X, Y=Y, k=k, N=eps_r,c=xy_c, R=R_sc, U=1+0j,theta_inc=theta_inc, A=A, B=B)
    Z_FEM = u_TOT(Omega(X.flatten(),Y.flatten())).reshape(X.shape)
    import matplotlib.pyplot as plt 
    from matplotlib.tri import Triangulation
    from matplotlib.patches import Circle, Rectangle

    triangles = np.array([[v.nr for v in T.vertices] for T in Omega.Elements()])
    tri = Triangulation(x_points, y_points, triangles)

    print(f'{z.shape=}')
    print(f'{x_points.shape=}')
    print(f'{len(tri.x)=}')
    
    fig, ax = plt.subplots(ncols=2,nrows=2)
    
    ax[0,0].tripcolor(tri, np.real(z)[:,0])
    ax[0,0].add_patch(Circle(xy=[0,0], radius=R_PML, facecolor='none', edgecolor='r',linestyle='--'))
    ax[0,0].add_patch(Rectangle(xy=(-L,-L), width=2*L, height=2*L,facecolor='none', edgecolor='w',linestyle='--'))
    ax[0,0].axis('square')
    ax[0,1].pcolormesh(X,Y,np.real(Z))
    ax[0,1].axis('square')
    ax[1,0].pcolormesh(X,Y,np.real(Z_FEM))
    ax[1,0].axis('square')
    ax[1,1].pcolormesh(X,Y,np.abs(Z_FEM-Z))
    ax[1,1].axis('square')
    print(np.linalg.norm(Z_FEM - Z))
    print(np.linalg.norm(Z_FEM - Z, np.inf))
    

    plt.show()


  

def test_far_field_from_plane_wave():
    geo = SplineGeometry()
    
    landa = 2*np.pi/k
    R_PML = np.sqrt(x_sc**2 + y_sc**2) + R_sc + 2*landa
    delta_PML = 3*landa
    
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
    Omega = Mesh(geo.GenerateMesh())
    
    Omega.Curve(3)
    alpha = 1E-2
    # the alpha probably will need to be adjusted
    Omega.SetPML(pml.Radial(rad=R_PML,alpha=alpha*1j,origin=(0,0)),"PML")

    # variational formulation
    polynomial_order = 4
    n_values = { "scatterer" : eps_r, "background" : 1, "PML" : 1}
    n = Omega.MaterialCF(n_values) 

    N_R = 72
    theta_r = np.linspace(0, 2*np.pi, N_R, endpoint=False)
    x_hat = np.column_stack([np.cos(theta_r), np.sin(theta_r)])

    N_E = 36
    theta_E = np.linspace(0, 2*np.pi, N_E, endpoint= False) + np.pi
    d_inc = np.column_stack([np.cos(theta_E), np.sin(theta_E)])
    N = specialcf.normal(Omega.dim)


    H = H1(Omega, order = polynomial_order, complex=True)
    u = H.TrialFunction()
    v = H.TestFunction()

    a = BilinearForm(H)
    a += SymbolicBFI(grad(u)*grad(v) - n*k**2*u*v)
    a += SymbolicBFI(-1j*k*u*v,definedon=Omega.Boundaries("outerbnd"))


    a.Assemble()
    A_inv = a.mat.Inverse()

    d = np.array([np.cos(theta_inc), np.sin(theta_inc)])
#    phi_0 = np.angle(U_inc[f_ID,(36+2*e)//72,e]/np.exp( 1j*k*(R_R)))
    # u_inc_ns = U_inc[f_ID,(36+2*e)%72,e]/np.exp( 1j*k*(R_R))*ns.exp( 1j*k*( d[0]*ns.x + d[1]*ns.y) ) # fitted plane wave 
    u_inc_ns = U*ns.exp( 1j*k*( d[0]*ns.x + d[1]*ns.y) )  #plane wave with U amplitude at origin
    l = LinearForm(H)
    l += SymbolicLFI((n-1)*k**2*u_inc_ns * v)
    l.Assemble()
    u_FEM = GridFunction(H)
    u_FEM.vec.data = A_inv * l.vec
    

    # ploting the FEM solution:
    vertices = np.array([ V.point for V in Omega.vertices])
    x_points = vertices[:,0]
    y_points = vertices[:,1]
    z = u_FEM(Omega(x_points, y_points))
    import matplotlib.pyplot as plt 
    from matplotlib.tri import Triangulation
    triangles = np.array([[v.nr for v in T.vertices] for T in Omega.Elements()])
    tri = Triangulation(x_points, y_points, triangles)

    print(f'{z.shape=}')
    print(f'{x_points.shape=}')
    print(f'{len(tri.x)=}')
    plt.tripcolor(tri, np.abs(z)[:,0])
    plt.axis('square')
    plt.show()


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
    FF = np.exp(1j*np.pi/4)/np.sqrt(8*np.pi*k)*(ff_u + ff_du)
    
    FF_exact = far_field_from_plane_wave( np.array([theta_inc]), theta_r, k, R_sc, xy_c, eps_r, NUMBER_OF_MODES)[:,0]
    print(np.abs(FF-FF_exact))

    assert_allclose(FF,FF_exact)
#    for e in range(N_E):
#        print(f'{f=} {e=}')
#        d = d_inc[e,:]
#        phi_0 = np.angle(U_inc[f_ID,(36+2*e)//72,e]/np.exp( 1j*k*(R_R)))
#        # u_inc_ns = U_inc[f_ID,(36+2*e)%72,e]/np.exp( 1j*k*(R_R))*ns.exp( 1j*k*( d[0]*ns.x + d[1]*ns.y) ) # fitted plane wave 
#        u_inc_ns = ns.exp( 1j*k*( d[0]*ns.x + d[1]*ns.y) )  #plane wave with (1 + 0j) amplitude at origin
#        l = LinearForm(H)
#        l += SymbolicLFI((n-1)*k**2*u_inc_ns * v)
#        l.Assemble()
#        u_FEM = GridFunction(H)
#        u_FEM.vec.data = A_inv * l.vec
#        FF[f,:,e] = compute_far_field(u_FEM)
#        #FF[:,e] = near_field_from_Hankel_expansion(u_FEM, Omega, R_PML, k) 



# def far_field_from_FEM(u_FEM, Omega, N, x_hat, k, polynomial_order = 4):
#     N_R, _ = x_hat.shape
#     ff_u = np.zeros(N_R, dtype=np.complex128)
#     ff_du = np.zeros(N_R, dtype=np.complex128)
#     for i in range(N_R):
#         d_r = x_hat[i,:]
#         E = ns.exp(-1J* k *( ns.x*d_r[0]+ ns.y*d_r[1]))
#         dphi = -1j*k*( d_r[0]*N[0] + d_r[1]*N[1] ) * E
#         H_out = H1(Omega, order=polynomial_order, complex=True, definedon=Omega.Materials("background"))
#         trace_E = GridFunction(H_out)
#         trace_E.vec[:] = 0
#         trace_E.Set(E, BND, definedon=Omega.Boundaries("scatterer"))
#         ff_u[i] = Integrate( dphi * u_FEM, Omega, order= polynomial_order+1, definedon=Omega.Boundaries("scatterer"))
#         ff_du[i] = Integrate(grad(trace_E)*grad(u_FEM)-k**2*trace_E*u_FEM, Omega, order= polynomial_order+1, definedon= Omega.Materials("background"))
# 
#     return np.exp(1j*np.pi/4)/np.sqrt(8*np.pi*k)*(ff_u + ff_du)

# def approx_near_field(u_FEM):
#     ff_u = np.zeros(N_R, dtype=np.complex128)
#     ff_du = np.zeros(N_R, dtype=np.complex128)
#     for i in range(N_R):
#         d_r = x_hat[i,:]
#         E = ns.exp(-1J* k *( ns.x*d_r[0]+ ns.y*d_r[1]))
#         dphi = -1j*k*( d_r[0]*N[0] + d_r[1]*N[1] ) * E
#         H_out = H1(Omega, order=polynomial_order, complex=True, definedon=Omega.Materials("background"))
#         trace_E = GridFunction(H_out)
#         trace_E.vec[:] = 0
#         trace_E.Set(E, BND, definedon=Omega.Boundaries("scatterer"))
#         ff_u[i] = np.exp(1j*k*R_R)/np.sqrt(R_R)*Integrate( dphi * u_FEM, Omega, order= polynomial_order+1, definedon=Omega.Boundaries("scatterer"))
#         ff_du[i] = np.exp(1j*k*R_R)/np.sqrt(R_R)*Integrate(grad(trace_E)*grad(u_FEM)-k**2*trace_E*u_FEM, Omega, order= polynomial_order+1, definedon= Omega.Materials("background"))
#     return np.exp(1j*np.pi/4)/np.sqrt(8*np.pi*k)*(ff_u + ff_du)

#def compute_near_field(u_FEM): ##!! NEEDS HANKEL FUNCTIONS IN NGSOLVE
#    ff_u = np.zeros(N_R, dtype=np.complex128)
#    ff_du = np.zeros(N_R, dtype=np.complex128)
#    for i in range(N_R):
#        d_r = x_hat[i,:]
#        E = ns.exp(-1J* k *( ns.x*d_r[0]+ ns.y*d_r[1]))
#        dphi = -1j*k*( d_r[0]*N[0] + d_r[1]*N[1] ) * E
#        H_out = H1(Omega, order=polynomial_order, complex=True, definedon=Omega.Materials("background"))
#        trace_E = GridFunction(H_out)
#        trace_E.vec[:] = 0
#        trace_E.Set(E, BND, definedon=Omega.Boundaries("scatterer"))
#        ff_u[i] = np.exp(1j*k*R_R)/np.sqrt(R_R)*Integrate( dphi * u_FEM, Omega, order= polynomial_order+1, definedon=Omega.Boundaries("scatterer"))
#        ff_du[i] = np.exp(1j*k*R_R)/np.sqrt(R_R)*Integrate(grad(trace_E)*grad(u_FEM)-k**2*trace_E*u_FEM, Omega, order= polynomial_order+1, definedon= Omega.Materials("background"))
#
#    return np.exp(1j*np.pi/4)/np.sqrt(8*np.pi*k)*(ff_u + ff_du)


if __name__== "__main__":
    test_total_field()