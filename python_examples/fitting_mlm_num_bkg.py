import numpy as np
import cogent_utilities as cu
import matplotlib.pylab as plt
import lichen.lichen as lch

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
def mygauss(x,mu,sigma):
    exponent = ((x-mu)**2)/(2*sigma**2)
    a = 1.0/(sigma*np.sqrt(2*np.pi))
    ret = a*np.exp(-exponent)
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

    tot_events = abs(p[3])
    num_bkg = abs(p[4])
    num_sig = tot_events - num_bkg
    


    ret = num_sig*mygauss(x,p[0],p[1]) + num_bkg*mylinear(x,p[2],xlo,xhi)
    return ret

################################################################################
def negative_log_likelihood(p, x, fixed_parameters):
    # Here you need to code up the sum of all of the negative log likelihoods (pdf)
    # for each data point.
    
    tot_events = abs(p[3])
    num_bkg = abs(p[4])
    num_sig = tot_events - num_bkg

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
infile_name = '/Users/lm27apic/Documents/Fall_2012/Dark_Matter_Research/dark_matter_data/low_gain.txt'

tdays,energies = cu.get_cogent_data(infile_name,first_event=first_event,calibration=999)

xlo = 0.133
xhi = 0.1470
mu = (xlo + xhi) / 2
sigma = 0.0085
slope = 0.5


# Index the range of energies
index0 = energies>xlo
index1 = energies<xhi
index = index0*index1

x = energies[index]

plt.figure()
lch.hist_err(x,bins=50)


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
params_starting_vals = [mu, sigma, slope, npts, 0.05*npts]
fixed_parameters = [xlo,xhi,npts]
params_final_vals = optimize.fmin(negative_log_likelihood, params_starting_vals[:],args=(x,fixed_parameters),full_output=True,maxiter=10000)

print params_final_vals

print "Final values"
print params_final_vals[0][3]
print params_final_vals[0][4]
print params_final_vals[0][3] - params_final_vals[0][4]

plt.show()



