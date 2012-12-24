import numpy as np
import cogent_utilities as cu
import matplotlib.pylab as plt
import lichen.lichen as lch

import scipy.stats as stats

from scipy import optimize
from scipy import integrate

################################################################################
def mylinear(x,m,xlo,xhi):

    npts = len(x)
    num_int_points = 1000

    y = np.ones(npts)

    ############################################################################
    # Check the normalization
    ############################################################################
    val0 = slope*xlo + 1
    val1 = slope*xhi + 1

    minval = val0
    maxval = val1
    if val1<val0:
        minval = val1
        maxval = val0

    xranges = xhi-xlo
    ############################################################################

    xnorm = np.linspace(xlo,xhi,num_int_points)
    ynorm = np.ones(num_int_points)[0]

    y += m*x
    ynorm += m*xnorm

    # Does this keep the term from going negative?
    y     -= minval
    ynorm -= minval

    normalization = integrate.simps(ynorm,x=xnorm)

    ret = y/normalization
    return ret

################################################################################
def mygauss(x,mu,sigma,xlo,xhi):
    #exponent = ((x-mu)**2)/(2*sigma**2)
    #a = 1.0/(sigma*np.sqrt(2*np.pi))
    #ret = a*np.exp(-exponent)

    # Do we need this to keep track of the normalization?
    gauss_func = stats.norm(loc=mu,scale=sigma)
    xnorm = np.linspace(xlo,xhi,1000)
    ynorm = gauss_func.pdf(xnorm)
    normalization = integrate.simps(ynorm,x=xnorm)
    ret = gauss_func.pdf(x)/normalization

    return ret

################################################################################
def pdf(p,x,fixed_parameters): # Probability distribution function 
    # p is an array of the parameters 
    # x is the data points
    # So p[0] will be whatever you want.
    # fixed_parameters allows us to pass in some extra parameters for the fit.
    xlo = fixed_parameters[0]
    xhi = fixed_parameters[1]
    npts = fixed_parameters[2]

    num_sig = 0.0
    for i in range(0,11):
        index = i*3
        num_sig += abs(p[2+index])

    num_bkg = abs(p[-1])
    tot_events = num_sig + num_bkg

    ret = 0
    for i in range(0,11):
        index = i*3
        frac_gauss = p[2+index]/tot_events
        ret += frac_gauss*mygauss(x,p[0+index],p[1+index],xlo,xhi) 

    ret += (num_bkg/tot_events)*mylinear(x,p[-2],xlo,xhi)
    
    return ret

################################################################################
def negative_log_likelihood(p, x, fixed_parameters):
    # Here you need to code up the sum of all of the negative log likelihoods (pdf)
    # for each data point.
    num_sig = 0.0
    for i in range(0,11):
        index = i*3
        num_sig += abs(p[2+index])

    #num_sig = abs(p[2]) + abs(p[5]) + abs(p[8])
    num_bkg = abs(p[-1])
    tot_events = num_sig + num_bkg
    #tot_events = num_sig 

    npts = fixed_parameters[2]
    
    # Add in a Poisson term to constrain the number of events.
    mu = tot_events
    k = npts
    poisson_term = -mu + k*np.log(mu)

    ret = np.sum(-1*np.log(pdf(p,x,fixed_parameters))) - poisson_term
    return ret


################################################################################
# Importing the dark matter data
################################################################################
#
# This is the ``time" of the first event, in some digitial units used
# by the detector
first_event = 2750361.2
# Full path to the directory 
#infile_name = '/Users/lm27apic/Documents/Fall_2012/Dark_Matter_Research/dark_matter_data/low_gain.txt'
infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/low_gain.txt'
#infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/high_gain.txt'

tdays,energies = cu.get_cogent_data(infile_name,first_event=first_event,calibration=999)

# peaks go from high amplitued to low amplitude
xlo = [0.170,0.1566,0.1475,0.1330,0.120,0.106,0.106,0.106,0.97,0.088,0.080,0.073]
xhi = [0.178,0.1690,0.1565,0.1470,0.124,0.116,0.116,0.116,0.104,0.95,0.087,0.080]

index0 = energies>xlo[-1]
index1 = energies<xhi[0]
index = index0*index1

x = energies[index]

plt.figure()
lch.hist_err(x,bins=200)

################################################################################
# Now fit the data.
# Mean
# width
# slope
# number of events in Gaussian
# number of events in background
################################################################################
npts = len(x)
print "npts: ",npts

mu = np.ones(len(xlo))
sigma = np.ones(len(xlo))
nsig = np.ones(len(xlo))

# Signal parameters
for i in range(0,11):
    mu[i] = (xlo[i] + xhi[i]) / 2
    sigma[i] = 0.002
    nsig[i] = 0.08*npts

slope = 0.5
nbkg = 0.05*npts

params_starting_vals  = [mu[0],sigma[0],nsig[0]]
params_starting_vals += [mu[1],sigma[1],nsig[1]]
params_starting_vals += [mu[2],sigma[2],nsig[2]]
params_starting_vals += [mu[3],sigma[3],nsig[3]]
params_starting_vals += [mu[4],sigma[4],nsig[4]]
params_starting_vals += [mu[5],sigma[5],nsig[5]]
params_starting_vals += [mu[6],sigma[6],nsig[6]]
params_starting_vals += [mu[7],sigma[7],nsig[7]]
params_starting_vals += [mu[8],sigma[8],nsig[8]]
params_starting_vals += [mu[9],sigma[9],nsig[9]]
params_starting_vals += [mu[10],sigma[10],nsig[10]]

params_starting_vals += [slope,nbkg]


fixed_parameters = [xlo[-1],xhi[0],npts]
params_final_vals = optimize.fmin(negative_log_likelihood, params_starting_vals[:],args=(x,fixed_parameters),full_output=True,maxiter=10000000,maxfun=100000)

print params_final_vals
print "Final values"
for i in range(0,11):
    index = i*3
    print params_final_vals[0][0+index],params_final_vals[0][1+index],params_final_vals[0][2+index]
print params_final_vals[0][-2],params_final_vals[0][-1]

for i in range(0,11):
    index = i*3
    m = params_final_vals[0][0+index]
    plt.plot([m,m],[0,600])

plt.show()

