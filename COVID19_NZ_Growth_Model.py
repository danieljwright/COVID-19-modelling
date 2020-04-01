# Using non-linear least squares to fit Richard's curve to coronavirus data in NZ
# Author: Daniel Wright
# Data: 30/03

import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import decimal as dc


# Creating the model (Richard's Curve)
def model(t, B):
    N = len(t)
    Y = np.zeros(N)
    for n in range(N):
        Y[n] = B[0]/((1+B[1]*np.exp(-B[2]*t[n]))**(1/B[3]))
    return Y


def model_nonlinear_least_squares_fit(t, Y, iterations=5):

    N = len(t)
    A = np.ones((N, 4))
    B = [6000, 76.46, 0.002474, 0.5]

    for i in range(iterations):
        # Calculate the Jacobians for current estimate of parameters
        # Currently does not converge
        for n in range(N):
            A[n, 0] = 1/((1+B[1]*np.exp(-B[2]*t[n]))**(1/B[3]))
            A[n, 1] = (-(B[0]*np.exp(-B[2]*t[n])/(B[3]*(1+B[1]*np.exp(-B[2]*t[n]))**(1/B[3] + 1))))
            A[n, 2] = (B[0]*t[n]*B[1]*np.exp(-B[2]*t[n]))/(B[3]*(1+B[1]*np.exp(-B[2]*t[n]))**(1/B[3]))
            A[n, 3] = (B[0]*np.log(1+B[1]*np.exp(-B[2]*t[n])))/((B[3]**2)*(1+B[1]*np.exp(-B[2]*t[n]))**(1/B[3]))

            # Use least squares to estimate the parameters
            deltaB, res, rank, s = lstsq(A, Y - model(t, B))
            B += deltaB
            print(B)
        return B

# Obtaining the data

a = np.genfromtxt('nz.csv', delimiter=",", skip_header=1)
positive = (a[:,0])
Y = np.cumsum(positive)
t = list(range(0, len(positive), 1))

# Calculating the parameters
B = model_nonlinear_least_squares_fit(t, Y)

# Extrapolating the curve
t1 = np.linspace(0, 56, 56)
Y1 = model(t1, B)


# Plotting
plt.figure(1)
plt.plot(t, Y, 'k')
plt.plot(t1, Y1, 'bo')
plt.show()
