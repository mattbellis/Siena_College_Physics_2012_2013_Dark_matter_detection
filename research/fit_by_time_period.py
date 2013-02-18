#IMPORTING ANYTHING THAT I MAY NEED
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


first_event = 2750361.2
start_date = datetime(2009, 12, 3, 0, 0, 0, 0)
ranges = [[8.0,13.0],[1.0,917.0]]
subranges = [[],[[1,68],[75,102],[108,306],[309,459],[551,917]]]

#########################################################
#IMPORT DATA
infile_name = '/Users/lm27apic/Documents/Dark_Matter_Research/dark_matter_data/low_gain.txt'
tdays,energies = cu.get_cogent_data(infile_name,first_event=first_event,calibration=0)
#########################################################

H_L = [80.3,270.8,270.8,244,6.075,77.233,271.74,70.86,999,312.01,27.7025,329]
t = [ 0, 68, 75, 102, 108, 306, 309, 460, 551, 917] 


for h_l in H_L:
    tau = h_l / np.log(2)

            
    frac_tot =  np.exp(-1/tau) - np.exp(-t[1]/tau)
    frac_tot += np.exp(-t[2]/tau) - np.exp(-100/tau)

    print "Lifetime:" , tau
    print "Time Period" , "0 to 100"
    print "Fraction Total:" , frac_tot

    frac_tot_before =  np.exp(-t[0]/tau) - np.exp(-t[1]/tau)
    frac_tot_before += np.exp(-t[2]/tau) - np.exp(-t[3]/tau)
    frac_tot_before += np.exp(-t[4]/tau) - np.exp(-t[5]/tau)
    frac_tot_before += np.exp(-t[6]/tau) - np.exp(-t[7]/tau)

    
    print "Time Period" , "0 to fire"
    print "Fraction Total:" , frac_tot_before


    frac_tot_after = np.exp(-t[8]/tau) - np.exp(-t[9]/tau)
        
    print "Time Period" , "fire to end"
    print "Fraction Total:" , frac_tot_after
    print "  "



    

        



            
