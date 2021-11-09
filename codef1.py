# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:03:16 2021

@author: clark
"""

import  numpy  as  np
import sys 
import pylab 
#import  data
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
        #if  i  <  10:
        #    print(xmass[i])
 
# make list of transverse momenta
transP = []
for i in range(0,nevent):
        transP.append(xdata[i][1])
        #if i < 10:
        #    print(transP[i])
            
# make list of rapidities
rapidity = []
for i in range(0,nevent):
        rapidity.append(xdata[i][1])
        #if i < 10:
        #    print(rapidity[i])

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
