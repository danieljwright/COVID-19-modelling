# Creating a model of coronavirus cases in NZ using polyfit
# Author: Daniel Wright
# Date: 28/03/2020

import pandas as pd
import numpy as np
import matplotlib.pyplot as p


# Load data
data1 = np.genfromtxt('nz.csv', delimiter=",", skip_header=1)

date = data1[:,1]
positive = (data1[:,0])
sum_pos = np.cumsum(positive)
length = len(date)
time = list(range(0, length, 1))

time = np.array(time, dtype=np.float32)

print('Time is: ', time)
print('Inv time is: ', inv_time)

# Calculate parameters for a 2nd order polynomial
coeff = np.polyfit(time, sum_pos, 2)

# Creat the model
model = np.poly1d(coeff)

# First and second derivatives of the model
d_model = np.polyder(model)
dd_model = np.polyder(d_model)


x = np.linspace(0, 56, 56)

# Calculate estimates of coronavirus cases
estimate = model(x)

v_estimate = d_model(x)
a_estimate = dd_model(x)


# Plotting
p.figure(2)
p.plot(time, sum_pos)
p.plot(x, Y)
p.title('Cases estimate using Richards Curve')

p.figure(1)
p.plot(time, sum_pos)
p.plot(x, estimate)
p.xlabel('Days')
p.ylabel('Positive Cases')
p.title('Number of positive cases in NZ')


##p.figure(2)
##p.plot(x, v_estimate)
##p.xlabel('Days')
##p.ylabel('Rate of change')
##p.title('First Dir (Rate of growth?)')
##
##p.figure(3)
##p.plot(x, a_estimate)
##p.xlabel('Days')
##p.ylabel('Acceleration')
##p.title('Second Dir (??)')

p.show()
