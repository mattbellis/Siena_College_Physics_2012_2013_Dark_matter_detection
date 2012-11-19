import numpy as np
import cogent_utilities as cu
import matplotlib.pylab as plt
import lichen.lichen as lch

# This is the ``time" of the first event, in some digitial units used
# by the detector
first_event = 2750361.2

# We need to give the full path to the directory. This will obviously be 
# different on your machine, so you will want to edit this by hand. 
infile_name = '/Users/lm27apic/Documents/Fall_2012/Dark_Matter_Research/dark_matter_data/low_gain.txt'
#infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/low_gain.txt'
#infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/high_gain.txt'

tdays,energies = cu.get_cogent_data(infile_name,first_event=first_event,calibration=999)

index0 = energies>0
index1 = energies<0.2

print len(energies)
print len(index0)
print len(index1)
index = index0*index1

x = energies[index]

# This will find the mean, sum, and stddev of an array z of some peak
index2 = energies>0.1330
index3 = energies<0.1470
index4 = index2*index3
print index4
z = energies[index4]
t = tdays[index4]

# plotting the peak
plt.figure()
#plt.hist(z)
lch.hist_err(z)
plt.xlabel('amplitude')
plt.ylabel('number of events')

# plotting the time for the peak
plt.figure()
plt.hist(t,bins=200)
plt.xlabel('time')
plt.ylabel('number of events')
# print np.polyfit(,,exp(x))


# background = 20*len(z)
mean = np.mean(z)
stdev = np.std(z)
summ = len(z)

print np.sum(z)
print mean
print stdev
print summ

# Make a figure on which to place the histogram
plt.figure()

# Note that we can set the number of bins (bins) and the x-axis range
# (range) in the constructor.

# plt.hist(energies,bins=100,range=(8,11))
# events = plt.hist(x,bins=300,range=(0,0.2))
lch.hist_err(x,bins=300,range=(0,0.2))
plt.xlabel('Amplitude')
plt.ylabel('Number of Events')
#plt.savefig('image~cal999_b200.png')

#plt.hist(x,bins=100,range=(0,900))

#print events

# Need to call the ``show" function to get the figure to pop up.
plt.show()
