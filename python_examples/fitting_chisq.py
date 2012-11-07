import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

################################################################################
def fitfunc(p, x):
    ret = p[1]*x + p[0]
    return ret

################################################################################
def errfunc(p, x, y, yerr):
    ret =  (((fitfunc(p, x)-y)**2)/yerr**2).sum()
    return ret

################################################################################
# Generate some fake data points
################################################################################
slope = 0.5
intercept = 4.0
x = np.array([1,2,3,4])
y = slope*x + intercept
y2 = slope*x + intercept

print x

# Add some random noise
yerr = np.array([0.1,0.3,0.2,0.25])
for i in range(0,len(y)):
    y[i] += np.random.normal(0,yerr[i])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(x,y,yerr=yerr,fmt='o')
ax.set_xlim(0,5)
ax.plot(x,y2,'-g')

params_starting_vals = [1.0,1.0]
params_final_vals = optimize.fmin(errfunc, params_starting_vals[:], args=(x,y,yerr),full_output=True)

print "Final values"
print params_final_vals
fit_intercept = params_final_vals[0][0]
fit_slope = params_final_vals[0][1]

print fit_intercept, fit_slope
y3 = fit_slope*x + fit_intercept
ax.plot(x,y3,'-r')

plt.show()
