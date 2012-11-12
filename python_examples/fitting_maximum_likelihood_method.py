import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

import lichen.lichen as lch

################################################################################
def mygauss(x,mu,sigma):
    exponent = ((x-mu)**2)/(2*sigma**2)
    a = 1.0/(sigma*np.sqrt(2*np.pi))
    ret = a*np.exp(-exponent)
    return ret

################################################################################
def pdf(p,x): # Probability distribution function 
    # p is an array of the parameters 
    # x is the data points
    # So p[0] will be whatever you want.
    # The functional form of your hypothesis (Gaussian).
    ret = mygauss(x,p[0],p[1])
    return ret

################################################################################
def negative_log_likelihood(p,x,y):
    # Here you need to code up the sum of all of the negative log likelihoods (pdf)
    # for each data point.
    ret = np.sum(-np.log(pdf(p,x)))
    return ret

################################################################################
# Generate some fake data points
################################################################################
mu = 5.0
sigma = 0.5
x = np.random.normal(mu,sigma,1000)
plt.figure()
lch.hist_err(x,bins=25)
print x

#plt.figure()
#prob = mygauss(x,mu,1.0)
#lch.hist_err(prob,bins=25)
#print prob
#plt.show()

# Now fit the data.
params_starting_vals = [1.0,1.0]
params_final_vals = optimize.fmin(negative_log_likelihood,params_starting_vals,args=(x,x),full_output=True,maxiter=100000)

print "Final values"
print params_final_vals
fit_mu = params_final_vals[0][0]
fit_sigma = params_final_vals[0][1]

# Matt will handle plotting the solution when we have the rest working.
#print fit_intercept, fit_slope
#y3 = mygauss(x,
#ax.plot(x,y3,'-r')

plt.show()
