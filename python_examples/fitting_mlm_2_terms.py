import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from scipy import integrate

import lichen.lichen as lch

################################################################################
def generate_bkg(xlo,xhi,npts,slope):
    
    x = np.zeros(npts)
    val0 = slope*xlo + 1
    val1 = slope*xhi + 1

    minval = val0
    maxval = val1
    if val1<val0:
        minval = val1
        maxval = val0

    xranges = xhi-xlo

    count = 0
    while count<npts:

        xmaybe = xranges*np.random.random() + xlo

        ymaybe = slope*xmaybe + 1.0

        ytest = maxval*np.random.random()

        if ytest<ymaybe:
            x[count] = xmaybe
            count += 1

    return x



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

    num_sig = p[3]
    num_bkg = p[4]
    tot_events = num_sig + num_bkg

    num_sig /= tot_events
    num_bkg /= tot_events

    ret = num_sig*mygauss(x,p[0],p[1]) + num_bkg*mylinear(x,p[2],xlo,xhi)
    return ret

################################################################################
def negative_log_likelihood(p, x, fixed_parameters):
    # Here you need to code up the sum of all of the negative log likelihoods (pdf)
    # for each data point.
    num_sig = p[3]
    num_bkg = p[4]
    tot_events = num_sig + num_bkg

    npts = fixed_parameters[2]
    
    # Add in a Poisson term to constrain the number of events.
    mu = tot_events
    k = npts
    poisson_term = -mu + k*np.log(mu)

    ret = np.sum(-1*np.log(pdf(p,x,fixed_parameters))) - poisson_term
    return ret

################################################################################
# Generate some fake data points
################################################################################
# ``Signal" from a Gaussian
mu = 5.0
sigma = 0.5
xsig = np.random.normal(mu,sigma,1000)

# ``Background" from some linear term
xlo = 0.0
xhi = 10.0
slope = 0.2
xbkg = generate_bkg(xlo,xhi,5000,slope)

# Combine the signal and background into one dataset.
x = xsig.copy()
x = np.append(x,xbkg)

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
params_starting_vals = [4.0, 0.5, 1.0, 0.5*npts, 0.5*npts]
fixed_parameters = [xlo,xhi,npts]
params_final_vals = optimize.fmin(negative_log_likelihood, params_starting_vals[:],args=(x,fixed_parameters),full_output=True,maxiter=10000)

print params_final_vals

print "Final values"
print params_final_vals[0][3]
print params_final_vals[0][4]
print params_final_vals[0][4] + params_final_vals[0][3]

plt.show()
