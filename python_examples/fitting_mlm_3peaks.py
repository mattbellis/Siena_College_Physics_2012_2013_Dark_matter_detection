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

    num_sig = abs(p[2]) + abs(p[5]) + abs(p[8])

    num_bkg = abs(p[-1])
    tot_events = num_sig + num_bkg
    #tot_events = num_sig 

    #num_sig /= tot_events
    #num_bkg /= tot_events

    ret = 0
    for i in range(0,3):
        index = i*3
        frac_gauss =  p[2+index]/tot_events
        ret += frac_gauss*mygauss(x,p[0+index],p[1+index]) 

    ret += (num_bkg/tot_events)*mylinear(x,p[-2],xlo,xhi)
    
    return ret

################################################################################
def negative_log_likelihood(p, x, fixed_parameters):
    # Here you need to code up the sum of all of the negative log likelihoods (pdf)
    # for each data point.
    num_sig = abs(p[2]) + abs(p[5]) + abs(p[8])
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
infile_name = '/Users/lm27apic/Documents/Fall_2012/Dark_Matter_Research/dark_matter_data/low_gain.txt'
#infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/low_gain.txt'
#infile_name = '/home/bellis/matts-work-environment/PyROOT/CoGeNT/data/high_gain.txt'

tdays,energies = cu.get_cogent_data(infile_name,first_event=first_event,calibration=999)

xlo1 = 0.1330
xhi1 = 0.1470

xlo2 = 0.1470
xhi2 = 0.1565

xlo3 = 0.1565
xhi3 = 0.1698

mu1 = (xlo1 + xhi1) / 2
mu2 = (xlo2 + xhi2) / 2
mu3 = (xlo3 + xhi3) / 2
sigma1 = 0.002
sigma2 = 0.002
sigma3 = 0.002
slope = 0.5

index0 = energies>xlo1
index1 = energies<xhi3
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

# Signal parameters
sigma = [sigma1,sigma2,sigma3]
mu = [mu1,mu2,mu3]
nsig = [0.3*npts, 0.3*npts, 0.3*npts]
nbkg = 0.05*npts

#params_starting_vals = [mu1, sigma1, mu2, sigma2, mu3, sigma3, slope, 0.95*npts, 0.05*npts]
params_starting_vals  = [mu[0],sigma[0],nsig[0]]
params_starting_vals += [mu[1],sigma[1],nsig[1]]
params_starting_vals += [mu[2],sigma[2],nsig[2]]
params_starting_vals += [slope,nbkg]

fixed_parameters = [xlo1,xhi3,npts]
params_final_vals = optimize.fmin(negative_log_likelihood, params_starting_vals[:],args=(x,fixed_parameters),full_output=True,maxiter=1000000,maxfun=10000)

print params_final_vals
print "Final values"
for i in range(0,3):
    index = i*3
    print params_final_vals[0][0+index],params_final_vals[0][1+index],params_final_vals[0][2+index]
print params_final_vals[0][-2],params_final_vals[0][-1]


plt.show()

