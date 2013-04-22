import numpy as np
import cogent_utilities as cu
import matplotlib.pylab as plt
import lichen.lichen as lch

# This is the ``time" of the first event, in some digitial units used
# by the detector
first_event = 2750361.2

# We need to give the full path to the directory. This will obviously be 
# different on your machine, so you will want to edit this by hand. 
infile_name = '/Users/lm27apic/Documents/Dark_Matter_Research/dark_matter_data/low_gain.txt'
#infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/low_gain.txt'
#infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/high_gain.txt'

tdays,energies = cu.get_cogent_data(infile_name,first_event=first_event,calibration=0)

index0 = energies>0.0
index1 = energies<13.0

print len(energies)
print len(index0)
print len(index1)
index = index0*index1
x = energies[index]

plt.figure()
lch.hist_err(x,bins=250,range=(0.5,12.0))
plt.xlabel('Energy (keVee)',fontsize=18)
plt.ylabel('Number of Events',fontsize=18)
#plt.annotate('The K-shell decays',xy=(10.8,295),xytext=(10.8,310),arrowprops=dict(arrowstyle='->'))
plt.xlim(0.5,12.0)
plt.yscale('log')
plt.ylim(10)
plt.savefig('fulldata.png')
plt.show()
