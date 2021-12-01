#%%
"""
Histogram Plotting Code
Created on Thu Nov  4 15:03:16 2021

@author: clark
"""
"""
This code plots the mass, transverse momentum of muon pair, rapidity, 
momentum of the muon pair, then the transverse momenta of each of the particles
No of bins is calculated using the Freeman-Diaconis rule
"""
import  numpy  as  np
import pylab
from scipy import stats
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


f  =  open("ups-15-small.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.array(np.split(datalist,nevent))


#%%
def plot_histogram(name, values, units, min, max)  :     
    #find binwidth, use Freeman-Diaconis rule
    mass_iqr = stats.iqr(xmass)
    bin_width = 2*mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)
    pylab.hist(values,  bins=num_bins,  range=[min, max])
    pylab.title("Histogram of " + name + " data" )
    pylab.ylabel("Counts in bin")
    pylab.xlabel(name + " " + units)
    pylab.xlim(np.min(values))
    pylab.show()

#%%
#  make  list  of  invariant  mass  of  events
xmass_name = r'$Mass ( \mu ^-\mu ^+)$ $(GeV/c^2)$'
xmass_units = r'Candidates/ (25 Mev$/c^2$)'

xmass = xdata[:,0]

plot_histogram(xmass_name, xmass, xmass_units, np.min(xmass), np.max(xmass))


#%%
#make list of transverse momenta of muon pair
trans_name = r'Transverse momentum of the muon pair'
mom_units = r'Candidates/ (25 Mev$/c^2$)'
trans_momentum_pair = xdata[:,1]

plot_histogram(trans_name, trans_momentum_pair, mom_units, 0, 30)


#%%          
#make list of rapidities of muon pair

rapidity_name = "Rapidity"
rapidity_units = "[absolute value]"

rapidity = xdata[:,2]

plot_histogram(rapidity_name, rapidity, rapidity_units, np.min(rapidity), np.max(rapidity))


#%%

mom_pair_name = "Momentum of muon pair"
momentum_pair = xdata[:,3]

plot_histogram(mom_pair_name, momentum_pair, mom_units, np.min(momentum_pair), 700)

#%%
trans_first_name = "Transverse momentum of first muon"
trans_momentum_first = xdata[:,4])

plot_histogram(trans_first_name, trans_momentum_first, mom_units, np.min(trans_momentum_first), 25)


#%%
trans_second_name = "Transverse momentum of second muon"
trans_momentum_second = xdata[:, 5]

plot_histogram(trans_second_name, trans_momentum_second, mom_units, np.min(trans_momentum_second), 25)
#make list of momenta of muon pair
