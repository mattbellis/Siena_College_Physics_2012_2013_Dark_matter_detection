import matplotlib.pylab as plt
import numpy as np

xlabels = [r'$^{49}$Ti',r'$^{51}$Va',r'$^{54}$Cr',r'$^{56}$Mn',r'$^{56,57,58}$Fe',r'$^{56}$Co',\
           r'$^{65}$Cu', r'$^{68}$Zn',r'$^{68}$Ga',r'$^{73}$Ge']

x0 = range(0,len(xlabels))
x1 = []
for x in x0:
    x1.append(x+0.1)

# Means of Gaussians for fit from 0-100 days.
y_means0 = [4.7884, 5.426, 5.908, 6.505, 6.975, 7.716, 8.903, 9.715, 10.4, 11.12]
y_means_err0= [0.01171, 0.06179, 0.04881, 0.01956, 0.02109, 0.03381, 0.006239, 0.00981, 0.00286, 0.0139]

# Means of Gaussians for fit from 0-460 days (before the fire).
y_means1 = [4.856, 5.437, 5.9, 6.432, 6.986, 7.805, 8.904, 9.715, 10.4, 11.1]
y_means_err1= [0.01669, 0.0861, 0.0203, 0.01791, 0.002236, 0.0436, 0.003515, 0.005701, 0.001739, 0.01613]

# Make the figure and set the border padding.
plt.figure(figsize=(15,7))
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.15)

plt.xticks(x0,xlabels)
plt.tick_params(axis='x', labelsize=24)

plt.errorbar(x0,y_means0,yerr=y_means_err0,fmt='o',label='0-100')
plt.errorbar(x1,y_means1,yerr=y_means_err1,fmt='o',label='0-460 (before fire)')

plt.xlim(x0[0]-1,x0[-1]+2)
ymin,ymax = plt.ylim()
plt.ylim(ymin-1,ymax+1)

plt.legend()

plt.show()
