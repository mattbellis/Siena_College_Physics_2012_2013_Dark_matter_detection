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

Xlo = [0.170,0.1566,0.1475,0.1330,0.120,0.106,0.106,0.106,0.97,0.088,0,0.073]
Xhi = [0.178,0.1690,0.1565,0.1470,0.124,0.116,0.116,0.116,0.104,0.95,0,0.080]
H_L = [80.3,270.8,270.8,244,6.075,77.233,271.74,70.86,999,312.01,27.7025,329]

# This will find the mean, sum, and stddev of an array z of some peak
index2 = energies>Xlo[3]
index3 = energies<Xhi[3]
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

##########################################
# plotting the time for the peak
plt.figure()
plt.hist(t,bins=50)
plt.xlabel('time')
plt.ylabel('number of events')

tau = H_L[3] / np.log(2)

t0 = 0
t1 = 68
#off
t2 = 74
t3 = 102
#off
t4 = 107
t5 = 306
#off
t6 = 308
t7 = 460
#off

f1 = np.exp(-t0/tau) - np.exp(-t1/tau)
f2 = np.exp(-t2/tau) - np.exp(-t3/tau)
f3 = np.exp(-t4/tau) - np.exp(-t5/tau)
f4 = np.exp(-t6/tau) - np.exp(-t7/tau)

##N1 = 
##N2 = 
##N3 = 
##N4 = 

frac_tot = f1 + f2 + f3 + f4
##N0 = (N1 + N2 + N3 + N4) / frac_tot
print "Lifetime:" , tau
print "Fraction Total:" , frac_tot
##print "Total Number of Atoms:" , N0

##########################################

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
