import numpy as np
import matplotlib.pylab as plt

# t0 = 0-100
# t1 = before fire
# t2 = after fire (fixed means)

x_t0 = [0,1,2,3]
n_t0 = [623,100,2012,500]
nerr_t0 = [62,10,201,50]

x_t1 = [0.1,1.1,2.1,3.1]
n_t1 = [634,95,2200,400]
nerr_t1 = [62,10,201,50]

plt.figure()
plt.errorbar(x_t0,n_t0,yerr=nerr_t0,fmt='o')
plt.errorbar(x_t1,n_t1,yerr=nerr_t1,fmt='o')

plt.xlim(-1,5)

plt.show()


