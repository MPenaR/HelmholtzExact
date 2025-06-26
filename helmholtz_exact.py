import numpy as np
from scipy.special import hankel1 as H1, jn as J, h1vp as dH1, jvp as dJ

import numpy.typing as npt

complex_array = npt.NDArray[np.complex128]
float_array = npt.NDArray[np.float64]


def Det(A_11: complex_array, A_21: complex_array,
        A_12: complex_array, A_22: complex_array) -> complex_array:
    """Computes the determinant of a two-by-two matrix given
    by its elements. Inputs are allowed to be arrays of the same shape"""
    return A_11 * A_22 - A_12*A_21


def DielectricPlaneWaveCoefficients(k: float, N: float, R: float, xy_c: float_array, U: complex, theta_inc: float, M: int) -> tuple[complex_array, complex_array]:
    """Computes the coefficientes of the Bessel expansion of the total field
    inside the scatterer and the scattered field outside.

    Inputs:
    - k: wavenumber of the background
    - N: index of refraction of the scatterer with respect to the background
    - c: center of the circular scatterer
    - R: radius of the circular scatterer
    - U: complex amplitude of the incident plane wave
    - theta_inc: angles that the propagating direction of the incident plane wave
    forms with the x-axis.
    - M: number of modes used in the expansion, i.e. n = -M, -M+1, ..., M-1, M
    Outputs:
    - A: coefficients of the scattered field outside the scatterer:
        u_s(r, theta) = sum_{n=-M}^M a_n H^1_n*(k*r)*exp(i*n*theta)
    - B: coefficients of the total field inside the scatterer:
        u(r, theta) = sum_{n=-M}^M b_n*J_n(k*r)*exp(i*n*theta)"""

    # vectorized version
    n = np.arange(-M, M+1, dtype=np.int64)
    dx = np.cos(theta_inc)
    dy = np.sin(theta_inc)
    W = Det(H1(n, k*R), dH1(n, k*R), -J(n, np.sqrt(N)*k*R), -np.sqrt(N)*dJ(n, np.sqrt(N)*k*R))
    A = -U*np.exp(1j*k*(dx*xy_c[0] + dy*xy_c[1]))*np.exp(-1j*n*theta_inc)*1j**n*Det(J(n, k*R), dJ(n, k*R), -J(n,np.sqrt(N)*k*R), -np.sqrt(N)*dJ(n,np.sqrt(N)*k*R))/W        
    B = -U*np.exp(1j*k*(dx*xy_c[0] + dy*xy_c[1]))*np.exp(-1j*n*theta_inc)*1j**n*Det(H1(n, k*R), dH1(n, k*R), J(n,k*R), dJ(n,k*R))/W
    
    return (A, B) 

def PlaneWave(X: float_array, Y: float_array, k: float, theta_inc: float = 0., U: complex = 1 + 0j) -> complex_array: 
    """
    Evaluates a plane wave field in all the (x,y) points given
    """
    dx = np.cos(theta_inc)
    dy = np.sin(theta_inc)
    return U*np.exp(1j*k*(dx*X + dy*Y))

def Fundamental(X: float_array,
                Y: float_array,
                k: float,
                x_s: float,
                y_s: float,
                U: complex = 1.+0.j) -> complex_array:
    """Evaluates a point source wave"""
    return U*1j/4  * H1(0, k*np.hypot(X-x_s, Y-y_s))

def U_tot_from_coeffs(X: float_array, Y: float_array, k: float, N: float,
                      c: float_array, R: float, U: complex,
                      theta_inc: float, A: complex_array, B: complex_array) -> complex_array:
    M = (len(A)-1)//2
    n = np.arange(-M, M+1, dtype=np.int64)
    r = np.hypot(X- c[0], Y- c[1])
    n = np.expand_dims(n, axis = np.arange(X.ndim).tolist())
    r = np.expand_dims(r, axis = -1)
    theta = np.arctan2(Y-c[1], X-c[0])
    theta = np.expand_dims( theta, axis = -1)
    U_in  = np.dot( J(n,np.sqrt(N)*k*r)*np.exp(1j*n*theta), B)
    U_inc = PlaneWave(X, Y, k, theta_inc, U)
    U_out = U_inc + np.dot(H1(n,k*r)*np.exp(1j*n*theta), A)
    r = np.squeeze(r)
    U_tot = np.where(r > R, U_out, U_in)
    return U_tot

def mear_field_plane_wave(xy_E: float_array, xy_R: float_array, k: float, R: float, c: float_array, M: int) -> complex_array:
    """I don't like this implementation, as if you emmit from a given point
    your incident field should not be a plane wave"""
    pass


def far_field_from_plane_wave(theta_E: float_array, theta_R: float_array, k: float, R: float, c: float_array, N: float, M: int) -> complex_array:
    N_E = len(theta_E)
    N_R = len(theta_R)
    FF = np.zeros((N_R, N_E), dtype=np.complex128)
    n = np.arange(-M, M+1)
    n = np.expand_dims(n, 0)
    theta = np.expand_dims(theta_R, -1)
    x_hat = np.cos(theta)
    y_hat = np.sin(theta)
    for j, theta_inc in enumerate(theta_E):
        A, _ = DielectricPlaneWaveCoefficients(k, N, R, c, 1., theta_inc, M)
        FF[:, j] = np.sqrt(2/np.pi/k)*np.dot(np.exp(1j*n*(theta - np.pi/2) - np.pi/4 - c[0]*x_hat - c[1]*y_hat), A)

    return FF





if __name__ == "__main__":
    pass
