import matplotlib.pylab as plt
import numpy as np

xlabels = [r'$^{49}$Ti',r'$^{51}$Va',r'$^{54}$Cr',r'$^{56}$Mn',r'$^{56,57,58}$Fe',r'$^{56}$Co',\
           r'$^{65}$Cu', r'$^{68}$Zn',r'$^{68}$Ga',r'$^{73}$Ge']
xlabels2 = [r'$^{51}$Va',r'$^{54}$Cr',r'$^{56}$Mn',r'$^{56,57,58}$Fe',r'$^{56}$Co',\
           r'$^{65}$Cu', r'$^{68}$Zn',r'$^{68}$Ga',r'$^{73}$Ge',r'$^{73}$As']

x0 = range(0,len(xlabels))
x1 = []
x2 = []
x3 = []
for x in x0:
    x1.append(x+0.1)
    x2.append(x+0.2)
    x3.append(x+0.3)

# Means of Gaussians for fit from 0-100 days.
y_means0 = [4.7884, 5.426, 5.908, 6.505, 6.975, 7.716, 8.903, 9.715, 10.4, 11.12]
y_means_err0= [0.01171, 0.06179, 0.04881, 0.01956, 0.02109, 0.03381, 0.006239, 0.00981, 0.00286, 0.0139]

# Means of Gaussians for fit from 0-460 days (before the fire).
y_means1 = [4.856, 5.437, 5.9, 6.432, 6.986, 7.805, 8.904, 9.715, 10.4, 11.1]
y_means_err1= [0.01669, 0.0861, 0.0203, 0.01791, 0.002236, 0.0436, 0.003515, 0.005701, 0.001739, 0.01613]

# Means of Guassians for fit from 0-920 days (entire dataset float)
y_means2 = [4.87,5.501,5.896,6.422,6.982,8.449,8.904,9.709,10.39,11.1]
y_means_err2 = [0.01579,0.111,0.02428,0.01452,0.02313,0.1638,0.00367,0.005512,0.001614,0.01579]

# Actual means of Events
y_means3 = [4.97,5.47,5.99,6.54,7.11,7.71,8.98,9.66,10.37,11.10]
y_means_err3 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]


plt.figure(figsize=(15,7))
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.15)

plt.xticks(x0,xlabels)
plt.tick_params(axis='x', labelsize=24)

plt.errorbar(x0,y_means0,yerr=y_means_err0,fmt='o',label='0-100 days')
plt.errorbar(x1,y_means1,yerr=y_means_err1,fmt='o',label='0-460 days')
plt.errorbar(x2,y_means2,yerr=y_means_err2,fmt='o',label='0-917 days')
plt.errorbar(x3,y_means3,yerr=y_means_err3,fmt='v',label='CoGeNT Data',color='m')

plt.xlim(x0[0]-1,x0[-1]+2)
ymin,ymax = plt.ylim()
plt.ylim(ymin-1,ymax+1)
plt.xlabel('Final State Isotope',fontsize=18)
plt.ylabel('Energy',fontsize=18)
legend_location = 'upper left'
plt.legend(loc=legend_location)
plt.savefig('mean_compare.png')
#####################################################

# Widths of Gaussians for fit from 0-100 days
y_widths0 = [0.04,0.15,0.15,0.0567,0.109,0.04993,0.1338,0.09675,0.1012,0.08714]
y_widths_err0 = [0.01021,0.09098,0.08934,0.01805,0.02018,0.01513,0.005137,0.01031,0.002182,0.01348]

# Widths of Gaussians for fit from 0-460 days (before the fire).
y_widths1 = [0.1057,0.1474,0.1302,0.1122,0.1242,0.05414,0.1274,0.09603,0.105,0.1004]
y_widths_err1 = [0.01281,0.0241,0.01712,0.0143,0.01781,0.02306,0.002862,0.005124,0.001356,0.01426]

# Widths of Gaussians for fit from 0-917 days (entire dataset float)
y_widths2 = [0.1055,0.15,0.1225,0.1248,0.1221,0.15,0.1264,0.09631,0.1062,0.08351]
y_widths_err2 = [0.01397,0.1057,0.02098,0.01335,0.02275,0.1062,0.003216,0.00546,0.001303,0.01864]

plt.figure(figsize=(15,7))
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.15)

plt.xticks(x0,xlabels)
plt.tick_params(axis='x', labelsize=24)

plt.errorbar(x0,y_widths0,yerr=y_widths_err0,fmt='o',label='0-100')
plt.errorbar(x1,y_widths1,yerr=y_widths_err1,fmt='o',label='0-460 (before fire)')
plt.errorbar(x2,y_widths2,yerr=y_widths_err2,fmt='o',label='0-917 (entire)')

plt.xlim(x0[0]-1,x0[-1]+2)
ymin,ymax = plt.ylim([0.0,0.5])
plt.ylim(ymin-1,ymax+1)
plt.xlabel('Nuclear Decays')
plt.ylabel('Widths')
plt.legend()
plt.savefig('width_compare.png')


#####################################################

xlabels = [r'$^{49}$Ti',r'$^{51}$Va',r'$^{54}$Cr',r'$^{56}$Mn',r'$^{56}$Fe',r'$^{57}$Fe',r'$^{58}$Fe',r'$^{56}$Co',\
           r'$^{65}$Cu', r'$^{68}$Zn',r'$^{68}$Ga',r'$^{73}$Ge']

x0 = range(0,len(xlabels))
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
for x in x0:
    x1.append(x+0.1)
    x2.append(x+0.2)
    x3.append(x+0.3)
    x4.append(x+0.4)
    x5.append(x+0.5)
    x6.append(x+0.3)
    x7.append(x+0.35)

# Number of events of Guassians for fit from 0-100 days
y_events0 = [112.9, 34.85,219.2,363.11,102.2,285.5,107.7,8.53,2432.7,701.2,6412.2,1213.1]
y_events_err0 = [32.98,10.75,52.79,107.7,17.77,49.63,18.73,4.91,107.92,65.80,177.63,18.07]

# Number of events of Gaussians for fit from 0-460 days (before fire)
y_events1 = [217.0,33.3,225.9,473.2,120.4,170.0,121.0,13.4,2390.6,685.4,6179.5,114.0]
y_events_err1 = [27.45,14.7,27.94,60.4,16.78,23.69,16.86,7.99,61.2,35.48,98.16,14.64]

# Number of events of Guassians for fit from 551-917 days (after fire 0-100)
y_events2 = [220.4,-100.0,226.8,83.1,-100.0,772.7,-100.0,-100.0,2186.8,652.3,6150.2,725.9]
#y_events2 = [220.4,12361516.0,226.8,83.1,9340.0,772.7,2193.0,6136000000000.0,2186.8,652.3,6150.2,725.9]
y_events_err2 = [76.03,100.0,69.19,11.15,100.0,445.15,100.0,100.0,147.19,86.51,213.70,1119.15]
#y_events_err2 = [76.03,11185617.0,69.19,11.15,5381.0,445.15,1263.0,1000000000.0,147.19,86.51,213.70,1119.15]

# Number of events of Guassians for fit from 551-917 days (after fire 0-fire)
y_events3 = [228.27,-100.0,252.6,776.4,-100.0,138.8,-100.0,-100.0,2199.4,662.1,6158.95,908.16]
#y_events3 = [228.27,14771622.9,252.6,776.4,4657.0,138.8,3015.3,11610000000000.0,2199.4,662.1,6158.95,908.16]
y_events_err3 = [64.21,100.0,76.51,98.62,100.0,67.4,100.0,100.0,147.34,86.98,213.79,1125.95]
#y_events_err3 = [64.21,11243926.0,76.51,98.62,22520.0,67.4,1458.0,10000000000.0,147.34,86.98,213.79,1125.95]

# Number of events of Guassians for fit from 0-917 days (entire dataset 0-100)
y_events4 = [212.9,40.9,238.5,575.3,115.2,133.3,115.4,12.43,2349.6,672.9,6164.5,105.9]
y_events_err4 = [42.03,23.72,35.94,62.87,22.3,25.81,22.35,18.42,58.24,35.64,90.02,20.44]

# Number of events of Guassians for fit from 0-917 days (entire dataset 0-fire)
y_events5 = [220.5,49.5,227.7,583.3,139.9,161.9,140.23,24.0,2355.7,676.9,6169.5,110.4]
y_events_err5 = [32.14,22.78,33.18,61.26,24.35,28.18,24.4,10.08,58.06,35.57,89.97,19.39]

# Number of events of Guassians for fit from 0-917 days (entire dataset float)
y_events6 = [221.4,52.4,215.4,587.4,141.0,163.2,141.3,33.2,2344.9,677.8,6175.6,112.1]
y_events_err6 = [31.93,25.91,38.59,61.83,24.75,28.65,24.81,20.89,60.97,35.51,90.02,20.41]

# Number of events from the CoGeNT 'How To'
y_events7 = [161.46,31.5,223.9,459.2,100.25,27.5,100.25,16.2,2117.8,520.15,6070.7,125.45]
y_events_pcterr7 =[12.263,15.238,9.33,11.63,44.88,381.09,44.88,23.45,2.23,5.114,1.35,33.48]
y_events_err7 = []
for val,pcterr in zip(y_events7,y_events_pcterr7):
    y_events_err7.append(val*pcterr/100)


plt.figure(figsize=(15,7))
plt.subplots_adjust(top=0.95,bottom=0.15,right=0.95,left=0.15)

plt.xticks(x0,xlabels)
plt.tick_params(axis='x', labelsize=24)

plt.errorbar(x0,y_events0,yerr=y_events_err0,fmt='o',label='0-100 days (float means)',color = 'b')
plt.errorbar(x1,y_events1,yerr=y_events_err1,fmt='o',label='0-460 days (float means)',color = 'g')
##plt.errorbar(x2,y_events2,yerr=y_events_err2,fmt='o',label='551-917 (0-100 means)',color = 'y')
##plt.errorbar(x3,y_events3,yerr=y_events_err3,fmt='o',label='551-917 (0-460 means)',color = 'c')
##plt.errorbar(x4,y_events4,yerr=y_events_err4,fmt='o',label='0-917 (0-100 means)',color = 'm')
##plt.errorbar(x5,y_events5,yerr=y_events_err5,fmt='o',label='0-917 (0-460 means)',color = 'k')
##plt.errorbar(x6,y_events6,yerr=y_events_err6,fmt='o',label='0-917 (float means)',color = 'r')
##plt.errorbar(x7,y_events7,yerr=y_events_err7,fmt='v',label='CoGeNT Results')
plt.errorbar(x2,y_events4,yerr=y_events_err4,fmt='o',label='0-917 days (fixed means to 0-100)',color = 'c')
plt.errorbar(x3,y_events5,yerr=y_events_err5,fmt='o',label='0-917 days (fixed means to 0-460)',color = 'y')
plt.errorbar(x4,y_events6,yerr=y_events_err6,fmt='o',label='0-917 days (float means)',color = 'r')
plt.errorbar(x5,y_events7,yerr=y_events_err7,fmt='v',label='CoGeNT Results',color = 'm')

plt.xlim(x0[0]-1,x0[-1]+2)
ymin,ymax = plt.ylim([-400,7000])
plt.ylim(ymin-1,ymax+1)
plt.xlabel('Final State Isotope',fontsize=18)
plt.ylabel('Number of Atoms Originally in Detector',fontsize=18)
legend_location = 'upper left'
plt.legend(loc=legend_location)
plt.savefig('event_compare.png')


plt.figure(figsize=(10,8))
plt.xticks(x0,xlabels)
plt.tick_params(axis='x', labelsize=24)
plt.errorbar(x0,y_events0,yerr=y_events_err0,fmt='o',label='0-100 days (float means)',color = 'b')
plt.errorbar(x1,y_events1,yerr=y_events_err1,fmt='o',label='0-460 days (float means)',color = 'g')
plt.errorbar(x2,y_events4,yerr=y_events_err4,fmt='o',label='0-917 days (fixed means to 0-100)',color = 'c')
plt.errorbar(x3,y_events5,yerr=y_events_err5,fmt='o',label='0-917 days (fixed means to 0-460)',color = 'y')
plt.errorbar(x4,y_events6,yerr=y_events_err6,fmt='o',label='0-917 days (float means)',color = 'r')
plt.errorbar(x5,y_events7,yerr=y_events_err7,fmt='v',label='CoGeNT Results',color = 'm')


plt.xlim(x0[-2]-0.1,x0[-2]+0.6)
ymin,ymax = plt.ylim([5800,6850])
plt.ylim(ymin-1,ymax+1)
plt.xlabel('Final State Isotope',fontsize=18)
plt.ylabel('Number of Atoms Originally in Detector',fontsize=18)
legend_location = 'upper right'
plt.legend(loc=legend_location)
plt.savefig('Ga_peak.png')

plt.show()
