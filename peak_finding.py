# -*- coding: utf-8 -*-
"""

Created on Thu Nov 11 14:16:35 2021

@author: Niamho

"""
#%%

import  numpy  as  np
from scipy import stats
from scipy import signal

#%%
f  =  open("ups-15-small.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.split(datalist,nevent)
print(nevent)
print(xdata[0])

#make list of masses
xmass  =  []
xmass_name = str("Mass")
xmass_units = str("[GeV/c^2]")
for  i  in  range(0,nevent):
        xmass.append(xdata[i][0])
#%%
#plot_histogram(xmass_name, xmass, xmass_units)
#find number of bins
mass_iqr = stats.iqr(xmass)
bin_width = 2*mass_iqr/((nevent)**(1/3))    
num_bins = int(2/bin_width)


#%%
#get count in each bin of the histogram
#count is a the list of counts in each bin
count, bin_edge, binNo = stats.binned_statistic(xmass, xmass, statistic = 'count', bins = num_bins, range=[np.min(xmass),np.max(xmass)])
bin_edge = bin_edge[:-1]

#take the bin centre to be the mean of all values that fall in the bin and therefore the value we're interested in
bin_centre = bin_edge + (bin_edge[1]-bin_edge[0])/2

#%%
#find peaks in the mass data
peaks_indices, peaks_prominences = signal.find_peaks(count, prominence=1000)
peak_vals = bin_centre[peaks_indices] #find the values of mass at histogram peaks
#find widths of peaks in data
widths,width_heights,left_ips,right_ips  = signal.peak_widths(count, peaks = peaks_indices, rel_height=1)
widths_mass = bin_width*widths #convert these widths to units of GeV/c^2
half_widths = (widths_mass/2) #half these widths and subtract and add to the peak centre in order to get range of peak
peak_range = (peaks_indices-half_widths, peaks_indices+half_widths)
range_peak_1 = (peak_vals[0]-half_widths[0], peak_vals[0]+half_widths[0])
##range_peak_1 is the range to be used for peak 1 
range_peak_2 = (peak_vals[1]-half_widths[1], peak_vals[1]+half_widths[1])
range_peak_3 = (peak_vals[2]-half_widths[2], peak_vals[2]+half_widths[2])

# %%
print(range_peak_1, range_peak_2, range_peak_3)

# %%
