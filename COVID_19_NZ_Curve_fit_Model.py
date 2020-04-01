# Script to fit Richard's Curve / Logisitic Curve to coronavirus cases in NZ
# Author: Daniel Wright
# Date: 31/03/2020

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


# Defining the model

#def model(t, B0, B1, B2, B3):
#    return B0/(1+B1*np.exp(-B2*t))**(1/B3)

def model(t, B0, B1, B2):
    return B0/(1+B1*np.exp(-B2*t))

# Importing data

data = np.genfromtxt('nz.csv', delimiter=",", skip_header=1)
positive1 = (data[:,0])
ydata1 = np.cumsum(positive1)[:(len(positive1)-3)] # data from 28/03
xdata1 = list(range(0, len(positive1), 1))[:(len(positive1)-3)]

print('ydata1: ', ydata1)
print('xdata1: ', xdata1)

positive2 = (data[:,3])
ydata2 = np.cumsum(positive2) # data from 31/03
xdata2 = list(range(0, len(positive2), 1))


# Finding the parameters

popt1, pcov1 = curve_fit(model, xdata1, ydata1)
popt2, pcov2 = curve_fit(model, xdata2, ydata2)
print(popt1)
print(popt2)

# Drawing the curve
t1 = np.linspace(0, 56, 56)
y1 = model(t1, popt1[0], popt1[1], popt1[2])
y2 = model(t1, popt2[0], popt2[1], popt2[2])


# Plotting
plt.figure(1)
plt.plot(xdata1, ydata1)
plt.plot(t1, y1)
plt.title('Model using 28/03 data')


plt.figure(2)
plt.plot(xdata2, ydata2)
plt.plot(t1, y2)
plt.title('Model using 31/03 data')


plt.figure(3)
plt.plot(xdata1, ydata1, 'bo')
plt.plot(xdata2, ydata2, 'k')
plt.title('Comparison of cases')

plt.show()



