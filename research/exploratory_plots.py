import numpy as np
import cogent_utilities as cu
import matplotlib.pylab as plt
import lichen.lichen as lch
import lichen.pdfs as pdfs
import lichen.iminuit_fitting_utilities as fitutils
import lichen.plotting_utilities as plotutils

import scipy.stats as stats

from cogent_utilities import *

from datetime import datetime,timedelta

import iminuit as minuit

#ranges = [[8.0,13.0],[1.0,917.0]]
#ranges = [[8.0,12.0],[1.0,459.0]]
#ranges = [[8.0,12.0],[551.0,917.0]]
#ranges = [[10.0,11.0],[551.0,917.0]]
#ranges = [[10.0,11.0],[1.0,459.0]]
ranges = [[10.2,10.6],[1.0,917.0]]
subranges = [[],[[1,68],[75,102],[108,306],[309,459],[551,917]]]
nbins = [15,30]
#nbins = [50,30]
bin_widths = np.ones(len(ranges))
for i,n,r in zip(xrange(len(nbins)),nbins,ranges):
    bin_widths[i] = (r[1]-r[0])/n

# This is the ``time" of the first event, in some digitial units used
# by the detector
first_event = 2750361.2
start_date = datetime(2009, 12, 3, 0, 0, 0, 0) #

################################################################################
# Importing the dark matter data
################################################################################
#
# Full path to the directory 
#infile_name = '/Users/lm27apic/Documents/Fall_2012/Dark_Matter_Research/dark_matter_data/low_gain.txt'
infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/low_gain.txt'
#infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/high_gain.txt'

tdays,energies = cu.get_cogent_data(infile_name,first_event=first_event,calibration=0)
data = [energies.copy(),tdays.copy()]

# Cut events out that fall outside the range.
data = cut_events_outside_range(data,ranges)
data = cut_events_outside_subrange(data,subranges[1],data_index=1)

nevents = float(len(data[0]))

############################################################################
# Plot the data
############################################################################
fig0 = plt.figure(figsize=(12,4),dpi=100)
ax0 = fig0.add_subplot(1,2,1)
ax1 = fig0.add_subplot(1,2,2)

ax0.set_xlim(ranges[0])
#ax0.set_ylim(0.0,50.0)
#ax0.set_ylim(0.0,92.0)
ax0.set_xlabel("Ionization Energy (keVee)",fontsize=12)
ax0.set_ylabel("Events/0.025 keVee",fontsize=12)

ax1.set_xlim(ranges[1])

ax1.set_xlabel("Days since 12/4/2009",fontsize=12)
ax1.set_ylabel("Event/30.6 days",fontsize=12)

lch.hist_err(data[0],bins=nbins[0],range=ranges[0],axes=ax0)
h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data[1],bins=nbins[1],range=ranges[1],axes=ax1)

# Do an acceptance correction of some t-bins by hand.
tbwidth = (ranges[1][1]-ranges[1][0])/float(nbins[1])
acc_corr = np.zeros(len(ypts))
# For 15 bins
#acc_corr[2] = tbwidth/(tbwidth-7.0)
#acc_corr[3] = tbwidth/(tbwidth-6.0)
#acc_corr[10] = tbwidth/(tbwidth-3.0)
# For 30 bins
#acc_corr[4] = tbwidth/(tbwidth-7.0)
#acc_corr[6] = tbwidth/(tbwidth-6.0)
#acc_corr[20] = tbwidth/(tbwidth-3.0)
# For 30 bins,full dataset
acc_corr[2] = tbwidth/(tbwidth-7.0)
acc_corr[3] = tbwidth/(tbwidth-6.0)
acc_corr[9] = tbwidth/(tbwidth-3.0)
ax1.errorbar(xpts, acc_corr*ypts,xerr=xpts_err,yerr=acc_corr*ypts_err,fmt='o', \
color='red',ecolor='red',markersize=2,barsabove=False,capsize=0)

############################################################################

fig1 = plt.figure(figsize=(12,6))
ax11 = fig1.add_subplot(1,2,1)
ax12 = fig1.add_subplot(1,2,2)

td = data[1]

master = None
for i in range(0,5):
    index0 = td>i*200
    index1 = td<=(i+1)*200
    index = index0*index1
    nevents = float(len(index[index]))
    print nevents
    h,xpts,ypts,xpts_err,ypts_err = lch.hist_err(data[0][index],bins=nbins[0],range=ranges[0],axes=ax11,color=(i/10,1,1))
    ypts = ypts.astype(float)
    print ypts
    ypts /= float(nevents)
    if i==0:
        master = ypts.copy()
    print ypts
    ypts /= master
    print ypts
    label = "%d-%d" % (i*200,(i+1)*200)
    ax12.plot(xpts,ypts,'o',markersize=15,label=label)


#ax12.set_ylim(-0.20,0.20)
ax12.set_ylim(0.20,2.20)
ax12.legend()


plt.show()

