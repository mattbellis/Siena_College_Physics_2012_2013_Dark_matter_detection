import numpy as np
import cogent_utilities as cu
import matplotlib.pylab as plt

#energy = [11.1031, 10.3671, 9.6586, 8.9789, 7.7089, 7.112, 7.112, 7.112, 6.539, 5.9892, 5.4651, 4.9664] 
#amp    = [0.173994, 0.163056, 0.152242, 0.139976, 0.122161, 0.110554, 0.110554, 0.110554, 0.100601, 0.091916, 0.08, 0.076519]
energy = [11.1031, 10.3671, 9.6586, 8.9789, 7.7089, 7.112, 7.112, 7.112, 6.539, 5.9892, 4.9664] 
amp    = [0.173994, 0.163056, 0.152242, 0.139976, 0.122161, 0.110554, 0.110554, 0.110554, 0.100601, 0.091916, 0.076519]

# Make a figure on which to place the histogram
plt.figure()

plt.plot(amp,energy,'bo')
plt.xlabel('Amplitude')
plt.ylabel('Energy (keVee)')

print np.polyfit(amp,energy,1)

# Need to call the ``show" function to get the figure to pop up.
plt.show()
