# -*- coding: utf-8 -*-
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
#  import  data
#xmass  =  np.loadtxt(sys.argv[1])



def plot_histogram(name, values, units)  :     
    #find binwidth, use Freeman-Diaconis rule
    mass_iqr = stats.iqr(xmass)
    bin_width = 2*mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)
    pylab.hist(values,  bins=num_bins,  range=[np.min(values), np.max(values)])
    pylab.title("Histogram of " + name + " data" )
    pylab.ylabel("Counts in bin")
    pylab.xlabel(name + " " + units)
    pylab.show()

f  =  open("ups-15-small.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.split(datalist,nevent)
print(nevent)
print(xdata[0])

#  make  list  of  invariant  mass  of  events
xmass  =  []
xmass_name = str("Mass")
xmass_units = str("[GeV/c^2]")
for  i  in  range(0,nevent):
        xmass.append(xdata[i][0])
 
plot_histogram(xmass_name, xmass, xmass_units)

#make list of transverse momenta of muon pair
trans_momentum_pair = []
trans_name = ("Transverse momentum of the muon pair")
mom_units = ("[GeV/c]")
for i in range(0,nevent):
        trans_momentum_pair.append(xdata[i][1])

plot_histogram(trans_name, trans_momentum_pair, mom_units)
            
#make list of rapidities of muon pair
rapidity = []
rapidity_name = "Rapidity"
rapidity_units = "[absolute value]"
for i in range(0,nevent):
        rapidity.append(xdata[i][2])

plot_histogram(rapidity_name, rapidity, rapidity_units)

momentum_pair = []
mom_pair_name = "Momentum of muon pair"
for i in range(0,nevent):
        momentum_pair.append(xdata[i][3])

plot_histogram(mom_pair_name, momentum_pair, mom_units)

trans_momentum_first = []
trans_first_name = "Transverse momentum of first muon"
for i in range(0, nevent):
    trans_momentum_first.append(xdata[i][4])

plot_histogram(trans_first_name, trans_momentum_first, mom_units)

trans_momentum_second = []
trans_second_name = "Transverse momentum of second muon"
for i in range(0, nevent):
        trans_momentum_second.append(xdata[i][5])      

plot_histogram(trans_second_name, trans_momentum_second, mom_units)            
#make list of momenta of muon pair

