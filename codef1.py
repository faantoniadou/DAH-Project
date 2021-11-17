# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:03:16 2021

@author: clark
"""

import  numpy  as  np
import sys 

import pylab
from scipy import stats
#  import  data

#xmass  =  np.loadtxt(sys.argv[1])

f  =  open("ups-15-small.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.split(datalist,nevent)
#print(nevent)
#print(xdata[0])


# make  list  of  invariant  mass  of  events
xmass  =  []
for  i  in  range(0,nevent):
        xmass.append(xdata[i][0])

#find binwidth, use Freeman-Diaconis rule
def plt_hist():
        mass_iqr = stats.iqr(xmass)
        bin_width = 2*mass_iqr/((nevent)**(1/3))
        #print(bin_width)
        num_bins = int(2/bin_width)
        #print(num_bins)
           
        pylab.hist(xmass,  bins=num_bins,  range=[np.min(xmass), np.max(xmass)])
        pylab.title("Histogram showing Upsilon(S1,S2,S3) peaks in mass spectrum")
        pylab.ylabel("Counts in bin")
        pylab.xlabel("Mass [GeV/c^2]")
        pylab.xlim(9,9.75)
        pylab.show()

#make list of transverse momenta of muon pair
trans__momentum_pair = []
for i in range(0,nevent):
        trans_momentum_pair.append(xdata[i][1])
        
            
#make list of rapidities of muon pair
rapidity = []
for i in range(0,nevent):
        rapidity.append(xdata[i][1])
        
            

def plt_hist():       
        pylab.hist(xmass, bins=123, range=[9, 9.75])
        pylab.title('Histogram of three mass peaks')
        pylab.xlabel(r'$M ( \mu ^-\mu ^+)$ $(GeV/c^2)$')
        pylab.ylabel(r'Candidates/ (25 Mev$/c^2$)')
        pylab.xlim(9,9.75)
        pylab.show()

# ask about order of tasks 1 & 2 

def get_peak1():
        region1 = xmass[np.where((xmass > 9.3) & (xmass < 9.6))]
        return region1


def fit_gaussian(xdata):


def fit_exp():

