# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:03:16 2021

@author: clark
"""

import  numpy  as  np
import sys 
#  import  data
#xmass  =  np.loadtxt(sys.argv[1])

f  =  open("ups-15-small.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.split(datalist,nevent)
print(nevent)
print(xdata[0])

#  make  list  of  invariant  mass  of  events
xmass  =  []
for  i  in  range(0,nevent):
        xmass.append(xdata[i][0])
        if  i  <  10:
            print(xmass[i])
 
#make list of transverse momenta
transP = []
for i in range(0,nevent):
        transP.append(xdata[i][1])
        if i < 10:
            print(transP[i])
            
#make list of rapidities
rapidity = []
for i in range(0,nevent):
        rapidity.append(xdata[i][1])
        if i < 10:
            print(rapidity[i])
            
#make list 
