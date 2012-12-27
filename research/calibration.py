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
# plt.show()



# Here we want to find the slope of the back ground if there is one.

x = [0.0328353, 0.0438772, 0.125795, 0.189068]
y = [23.9624, 22.0425, 17.0195, 13.9846]
slope = -61.55
intercept = 25.28
x2 = np.array([0.0328353, 0.0438772, 0.125795, 0.189068])
y2 = slope*x2 + intercept

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x,y,'bo')
ax.plot(x2,y2,'-g')
ax.axis([0,0.2,0,40])
print np.polyfit(x,y,1)

plt.show()
