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


def CircleDielectricCoeffs(k: float, N: float, R: float, c: float_array, theta_inc: float, M: int) -> tuple[complex_array, complex_array]:
    """Computes the coefficientes of the Bessel expansion of the total field
    inside the scatterer and the scattered field outside.

    Inputs: 
    - k: wavenumber of the background
    - N: index of refraction of the scatterer with respect to the background
    - c: center of the circular scatterer
    - R: radius of the circular scatterer
    - theta_inc: angles that the propagating direction of the incident field 
    forms with the x-axis. 
    - M: number of modes used in the expansion, i.e. n = -M, -M+1, ..., M-1, M
    Outputs:
    - A: coefficients of the scattered field outside the scatterer: 
        u_s(r, theta) = sum_{n=-M}^M a_n H^1_n*(k*r)*exp(i*n*theta)
    - B: coefficients of the total field inside the scatterer: 
        u(r, theta) = sum_{n=-M}^M b_n*J_n(k*r)*exp(i*n*theta)""" 

    # vectorized version
    n = np.arange(-M, M+1, dtype=np.int64)
    W = Det(H1(n,k*R), dH1(n,k*R), -J(n,np.sqrt(N)*k*R), -np.sqrt(N)*dJ(n,np.sqrt(N)*k*R))
    print(W.shape)
    A = -np.exp(-1j*n*theta_inc)*1j**n*Det(J(n,k*R), dJ(n,k*R), -J(n,np.sqrt(N)*k*R), -np.sqrt(N)*dJ(n,np.sqrt(N)*k*R))/W        
    B = -np.exp(-1j*n*theta_inc)*1j**n*Det(H1(n,k*R), dH1(n,k*R), J(n,k*R), dJ(n,k*R))/W
    # serial version
#    A = np.zeros(2*M+1, dtype=np.complex128)
#    B = np.zeros(2*M+1, dtype=np.complex128)
#
#    for n in range(-M, M+1):
#        i = n + M
#        W = Det(H1(n,k*R), dH1(n,k*R), -J(n,np.sqrt(N)*k*R), -np.sqrt(N)*dJ(n,np.sqrt(N)*k*R))
#        A[i] = -np.exp(-1j*n*theta_inc)*1j**n*Det(J(n,k*R), dJ(n,k*R), -J(n,np.sqrt(N)*k*R), -np.sqrt(N)*dJ(n,np.sqrt(N)*k*R))/W        
#        B[i] = -np.exp(-1j*n*theta_inc)*1j**n*Det(H1(n,k*R), dH1(n,k*R), J(n,k*R), dJ(n,k*R))/W
    
    return (A, B) 


if __name__ == "__main__":
    A, B = CircleDielectricCoeffs(1, 1, 1, np.array([0., 0.]), 0., 10)
    print(A)
