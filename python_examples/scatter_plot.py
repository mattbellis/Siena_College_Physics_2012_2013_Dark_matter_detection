import numpy as np
import matplotlib.pylab as plt

# Generate npts random numbers drawn from a normal (Gaussian)
# distribution with mean=5.0 and sigma=1.0
npts = 1000
mu = 5.0
sig = 1.0
# x will be a numpy array which has particular properties.
x = np.random.normal(loc=mu,scale=sig,size=npts)

# Do the same for npts more random numbers for a y-variable
y = np.random.normal(loc=mu,scale=sig,size=npts)

# Make a figure on which to place the histogram
plt.figure()

# Note that we can set the number of bins (bins) and the x-axis range
# (range) in the constructor.
plt.scatter(x,y,marker='o',s=5.0,c='blue')

# Need to call the ``show" function to get the figure to pop up.
plt.show()
