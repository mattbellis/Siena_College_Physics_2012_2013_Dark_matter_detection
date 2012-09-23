import numpy as np
import matplotlib.pylab as plt

# Generate 10000 random numbers between 1,5
# x will be a numpy array which has particular properties.
x = 1.0 + 4.0*np.random.random(10000)

# Make a figure on which to place the histogram
plt.figure()

# Note that we can set the number of bins (bins) and the x-axis range
# (range) in the constructor.
plt.hist(x,bins=100,range=(0,6))

# Need to call the ``show" function to get the figure to pop up.
plt.show()
