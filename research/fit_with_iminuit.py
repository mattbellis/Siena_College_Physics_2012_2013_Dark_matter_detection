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

ranges = [[8.0,13.0],[1.0,917.0]]
#ranges = [[8.0,12.0],[1.0,459.0]]
#ranges = [[8.0,12.0],[551.0,917.0]]
#ranges = [[10.0,11.0],[551.0,917.0]]
#ranges = [[10.0,11.0],[1.0,459.0]]
subranges = [[],[[1,68],[75,102],[108,306],[309,459],[551,917]]]
nbins = [300,30]
#nbins = [50,30]
bin_widths = np.ones(len(ranges))
for i,n,r in zip(xrange(len(nbins)),nbins,ranges):
    bin_widths[i] = (r[1]-r[0])/n

# This is the ``time" of the first event, in some digitial units used
# by the detector
first_event = 2750361.2
start_date = datetime(2009, 12, 3, 0, 0, 0, 0) #

################################################################################
# CoGeNT fit
################################################################################
def fitfunc(data,p,parnames,params_dict):

    pn = parnames

    flag = p[pn.index('flag')]

    tot_pdf = np.zeros(len(data[0]))
    pdf = None
    num_wimps = 0

    x = data[0]
    y = data[1]

    xlo = params_dict['var_e']['limits'][0]
    xhi = params_dict['var_e']['limits'][1]
    ylo = params_dict['var_t']['limits'][0]
    yhi = params_dict['var_t']['limits'][1]

    tot_pdf = np.zeros(len(x))

    num_flat = p[pn.index('num_flat')]
    e_exp0 = p[pn.index('e_exp0')]

    ############################################################################
    # k-shell peaks
    ############################################################################
    means = []
    sigmas = []
    numks = []
    decay_constants = []

    npeaks = 4

    num_tot = 0.0
    for name in parnames:
        if 'num_' in name or 'ncalc' in name:
            num_tot += p[parnames.index(name)]
        elif 'num_flat' in name or 'num_exp1' in name:
            num_tot += p[parnames.index(name)]

    print "num_tot: ",num_tot

    for i in xrange(npeaks):
        name = "ks_mean%d" % (i)
        means.append(p[pn.index(name)])
        name = "ks_sigma%d" % (i)
        sigmas.append(p[pn.index(name)])
        name = "ks_ncalc%d" % (i)
        numks.append(p[pn.index(name)]/num_tot) # Normalized this
                                                # to number of events.
    #name = "ls_dc%d" % (i)
    #decay_constants.append(p[pn.index(name)])

    for n,m,s in zip(numks,means,sigmas): 
        pdf  = pdfs.gauss(x,m,s,xlo,xhi)
        #dc = -1.0*dc
        #pdf *= pdfs.exp(y,dc,ylo,yhi,subranges=subranges[1])
        pdf *= n
        tot_pdf += pdf

    num_flat /= num_tot

    ############################################################################
    # Flat term
    ############################################################################
    #pdf  = pdfs.poly(x,[],xlo,xhi)
    pdf  = pdfs.exp(x,e_exp0,xlo,xhi)
    #pdf *= pdfs.poly(y,[],ylo,yhi,subranges=subranges[1])
    pdf *= num_flat
    #print "flat pdf: ",pdf[0:8]/num_flat
    #print "flat pdf: ",pdf[0:8]
    tot_pdf += pdf

    return tot_pdf







################################################################################
# Extended maximum likelihood function for minuit, normalized already.
################################################################################
def emlf_normalized_minuit(data,p,parnames,params_dict):

    ndata = len(data[0])

    flag = p[parnames.index('flag')]

    wimp_model = None

    num_tot = 0.0
    for name in parnames:
        if 'num_' in name or 'ncalc' in name:
            num_tot += p[parnames.index(name)]
        elif 'num_flat' in name or 'num_exp1' in name:
            num_tot += p[parnames.index(name)]

    tot_pdf = fitfunc(data,p,parnames,params_dict)

    likelihood_func = (-np.log(tot_pdf)).sum()

    print num_tot,ndata

    ret = likelihood_func - fitutils.pois(num_tot,ndata)

    return ret

################################################################################




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

#means = [11.10,10.37,9.66,8.98]
means = [11.11,10.4,9.70,8.91]
sigmas = [0.08,0.08,0.08,0.08]
num_decays_in_dataset = [100,4600,400,1400]

############################################################################
# Declare the fit parameters
############################################################################
params_dict = {}
#params_dict['flag'] = {'fix':True,'start_val':args.fit}
params_dict['flag'] = {'fix':True,'start_val':0}
params_dict['var_e'] = {'fix':True,'start_val':0,'limits':(ranges[0][0],ranges[0][1])}
params_dict['var_t'] = {'fix':True,'start_val':0,'limits':(ranges[1][0],ranges[1][1])}

for i,val in enumerate(means):
    name = "ks_mean%d" % (i)
    params_dict[name] = {'fix':True,'start_val':val}
for i,val in enumerate(sigmas):
    name = "ks_sigma%d" % (i)
    params_dict[name] = {'fix':False,'start_val':val,'limits':(0.001,0.15)}
for i,val in enumerate(num_decays_in_dataset):
    name = "ks_ncalc%d" % (i)
    params_dict[name] = {'fix':False,'start_val':val,'limits':(1.0,6000.0)}

params_dict['num_flat'] = {'fix':False,'start_val':1900.0,'limits':(0.0,5000.0)}
params_dict['e_exp0'] = {'fix':False,'start_val':0.5,'limits':(0.0,1.0)}

#plt.show()

params_names,kwd = fitutils.dict2kwd(params_dict)

f = fitutils.Minuit_FCN([data],params_dict,emlf_normalized_minuit)

m = minuit.Minuit(f,**kwd)

# For maximum likelihood method.
m.up = 0.5

# Up the tolerance.
m.tol = 1.0

m.migrad()

values = m.values







################################################################################
############################################################################
# Flat
############################################################################
expts = np.linspace(ranges[0][0],ranges[0][1],1000)
eytot = np.zeros(1000)


# Energy projections
#ypts = np.ones(len(expts))
ypts = np.exp(-values['e_exp0']*expts)
y,plot = plotutils.plot_pdf(expts,ypts,bin_width=bin_widths[0],scale=values['num_flat'],fmt='m-',axes=ax0)
eytot += y

# K-shell peaks
for i,meanc in enumerate(means):
    name = "ks_mean%d" % (i)
    m = values[name]
    name = "ks_sigma%d" % (i)
    s = values[name]
    name = "ks_ncalc%d" % (i)
    n = values[name]

    gauss = stats.norm(loc=m,scale=s)
    eypts = gauss.pdf(expts)

    # Energy distributions
    y,plot = plotutils.plot_pdf(expts,eypts,bin_width=bin_widths[0],scale=n,fmt='r-',axes=ax0)
    eytot += y
    #lshell_totx += y


ax0.plot(expts,eytot,'b',linewidth=2)









plt.show()

