################################################################################
# This example will read in the CoGeNT data and calibrate it so we work 
# in units of days and energies (keV)
# 
# I've already written a bunch of helper functions and put them in 
# cogent_utilities.py, which you see we import at the very beginning.
################################################################################

import numpy as np
import cogent_utilities as cu

# This is the ``time" of the first event, in some digitial units used
# by the detector
first_event = 2750361.2

# We need to give the full path to the directory. This will obviously be 
# different on your machine, so you will want to edit this by hand. 
#infile_name = '/Users/lm27apic/Documents/Fall_2012/Dark_Matter_Research/dark_matter_data/high_gain.txt'
infile_name = '/home/bellis/matts-work-environment/python/CoGeNT/data/before_fire_HG.dat'
#infile_name = '/home/bellis/matts-work-environment/python/CoGeNT/data/before_fire_LG.dat'
tdays,energies = cu.get_cogent_data(infile_name,first_event=first_event,calibration=0)

print "\nNumber of entries in the data arrays"
print "# time in days:    %d" % (len(tdays))
print "# energies in keV: %d" % (len(energies))

# And just for the heck of it, we can dump the first 5 entries of each array.
print "\nFirst five entries in arrays."
#print tdays[0:5]
#print energies[0:5]
for t,e in zip(tdays,energies):
    #if e<3.1:
    if e<12.0:
        print "%f,%f" % (t,e)

    


print "\n"
