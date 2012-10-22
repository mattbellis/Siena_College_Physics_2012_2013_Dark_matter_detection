import numpy as np
import cogent_utilities as cu
import matplotlib.pylab as plt

# This is the ``time" of the first event, in some digitial units used
# by the detector
first_event = 2750361.2

# We need to give the full path to the directory. This will obviously be 
# different on your machine, so you will want to edit this by hand. 
#infile_name = '/Users/lm27apic/Documents/low_gain.txt'
#infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/low_gain.txt'
infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/high_gain.txt'
tdays,energies = cu.get_cogent_data(infile_name,first_event=first_event,calibration=0)

index0 = energies>0.0
index1 = energies<2.0
print len(energies)
print len(index0)
print len(index1)
index = index0*index1
print index
x = energies[index]
#x = tdays[index]



# Make a figure on which to place the histogram
plt.figure()

# Note that we can set the number of bins (bins) and the x-axis range
# (range) in the constructor.
# plt.hist(energies,bins=100,range=(8,11))
plt.hist(x,bins=100,range=(0,2))
#plt.hist(x,bins=100,range=(0,900))

# Need to call the ``show" function to get the figure to pop up.
plt.show()
