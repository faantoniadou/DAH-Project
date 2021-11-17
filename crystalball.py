# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 15:39:16 2021

@author: clark
"""

import  numpy  as  np
import pylab
from scipy import stats

from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.special import erf



f  =  open("ups-15-small.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.array(np.split(datalist,nevent))

#  make  list  of  invariant  mass  of  events
xmass = xdata[:,0]          
xmass_name = r'$Mass ( \mu ^-\mu ^+)$ $(GeV/c^2)$'
xmass_units = r'Candidates/ (25 Mev$/c^2$)'

low_side_edge1, low_peak_edge1, high_peak_edge1, high_side_edge1 = [9.0, 9.29, 9.61, 9.75]
low_side_edge2, low_peak_edge2, high_peak_edge2, high_side_edge2 = [9.75, 9.9, 10.15, 10.2]
low_side_edge3, low_peak_edge3, high_peak_edge3, high_side_edge3 = [10.19, 10.2, 10.5, 11]

edges1 = np.array((low_side_edge1, low_peak_edge1, high_peak_edge1, high_side_edge1))
edges2 = np.array((low_side_edge2, low_peak_edge2, high_peak_edge2, high_side_edge2))
edges3 = np.array((low_side_edge3, low_peak_edge3, high_peak_edge3, high_side_edge3))

# define each peak region
region1 = np.array(xmass)[np.where((xmass > low_side_edge1) & (xmass < high_side_edge1))]
region2 = np.array(xmass)[np.where((xmass > low_side_edge2) & (xmass < high_side_edge2))]
region3 = np.array(xmass)[np.where((xmass > low_side_edge3) & (xmass < high_side_edge3))]

all_edges = np.array((edges1, edges2, edges3), dtype=object)
all_regions = np.array([region1, region2, region3], dtype=object)



#%%

def get_bins(values):
    # find binwidth, use Freeman-Diaconis rule
    mass_iqr = stats.iqr(values)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)

    return num_bins


def hist_data(values, num_bins, normed=False):
    # renders histogram data
    values, bins = np.histogram(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed)

    return values, bins


def hist_area(values, normed=False):
    #finds area of the histogram
    get_bins(values)
    values, bins = hist_data(values, num_bins, normed)
    area = np.sum(np.diff(bins) * values)

    return area


#%%

def plot_histogram(name, values, units, num_bins, normed=False):     
        # plot
    values, bins, _ = pylab.hist(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed, label='histogram', alpha=0.5)

    pylab.title("Histogram of " + name + " data" )
    pylab.ylabel("Counts in bin")
    pylab.xlabel(name + " " + units)


plot_histogram(xmass_name, xmass, xmass_name, get_bins(xmass), False)


#%%
def bg_data(values, edges_array):
    '''
    use this to loop over peak edges and get bg region only
    '''

    low_side_edge, low_peak_edge, high_peak_edge, high_side_edge = edges_array

    # number of events
    nevents = len(values)

    # define sideband regions each be half as wide as the signal region 
    side_low = xmass[np.where((values < low_peak_edge) & (values > low_side_edge))]
    side_high = xmass[np.where((values < high_side_edge) & (values > high_peak_edge))]

    # concatenate these arrays to get a dataset for background
    bg_data = np.hstack((side_low, np.zeros(len(values)-(len(side_low)+len(side_high))),side_high))

    return bg_data



#%%

def bg_merge_data(region_array, edges_array):
    '''
    creates array of all background data
    '''

    n_peaks = len(region_array)
    all_bg = np.full(n_peaks, None)

    for i in range(n_peaks):
        all_bg[i] = bg_data(xmass, edges_array[i])

    # get rid of empty regions 
    all_bg = np.array(np.concatenate(all_bg))
    all_bg = all_bg[np.where(all_bg != 0)]

    return all_bg


#%%
def bg_hist(region_array, edges_array):
    # render histogram data
    all_bg = bg_merge_data(region_array, edges_array)
    num_bins = (get_bins(all_bg)) +20
    
    bg_counts, bg_masses = hist_data(all_bg, num_bins)


    return bg_masses[0:-1], bg_counts, all_bg

#print(bg_hist(all_regions, all_edges))
#print(len(bg_masses))
#print(len(bg_counts))


def exp_decay(t, a, b):
    return a * np.exp(-b*t)


#%%
def fit_bg(region_array, edges_array):
    '''
    this fits the background data to an exponential decay function
    returns the function variables a and b for a * np.exp(-b*t)
    '''

    bg_masses, bg_counts, all_bg = bg_hist(region_array, edges_array)

    bg_masses = bg_masses[np.where(bg_counts != 0)]
    bg_counts = bg_counts[np.where(bg_counts != 0)]

    popt, pcov = curve_fit(exp_decay,  bg_masses,  bg_counts, p0=[19000, 0.595628], maxfev=900000)
    print(popt)
    return popt[0], popt[1]

print(fit_bg(all_regions, all_edges))


#%%
def remove_bg(region_array, edges_array):
    #first we need to render histogram data for the whole of region1
    bg_masses, bg_counts, all_bg = bg_hist(region_array, edges_array)

    numbins = get_bins(all_bg)+20

    all_counts, all_masses = hist_data(xmass, numbins)

    # create set of points to fit  
    x = np.linspace(np.min(xmass), np.max(xmass),len(all_masses)-1)
    a, b = fit_bg(region_array, edges_array)
    clear_data = all_counts - exp_decay(x, a, b)

    return all_masses[0:-1], clear_data, exp_decay(x, a, b), x


#%%
def plot_bg():
    '''
    We can plot this stuff to visualise our cleared signal
    '''

    bg_masses, bg_counts, bg_all = bg_hist(all_regions, all_edges)

    numbins1 = get_bins(bg_all)+20

    _, clear_data, exp_fit, x = remove_bg(all_regions, all_edges)

    allcounts, all_masses = hist_data(xmass, numbins1)

    all_masses = all_masses[0:-1]
   # plot_histogram(xmass_name, bg_all[np.where(bg_all != 0)], xmass_units, numbins1, normed=False)
    pylab.plot(all_masses, clear_data, label='cleared signal')
    pylab.plot(all_masses, allcounts, label='all signals')
    pylab.plot(bg_masses, bg_counts, label='background data')
    pylab.plot(x, exp_fit, label='best fit curve')
    pylab.title("Graph of " + str(xmass_name) + " data" )
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    pylab.xlim(np.min(xmass), np.max(xmass))
    pylab.ylim(0)
    pylab.legend()
    pylab.show()

plot_bg()

##GAUSSIANS
#%%

def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

#%%

def fit_gaussian(region_array, edges_array):
    '''
    use this to loop over peaks and get bg region only
    '''

    all_masses, clear_data, exp_fit, x = remove_bg(all_regions, all_edges)      # need to extract all data

    # need to separate peaks

    _, low_peak_edge, high_peak_edge, _ = edges_array

    # take data outside of background and only within peaks
    peak_data = all_masses[np.where((all_masses >= low_peak_edge) & (all_masses <= high_peak_edge))]
    clear_data = clear_data[np.where((all_masses >= low_peak_edge) & (all_masses <= high_peak_edge))]

    '''
    This fits the peak to a normal distribution
    '''
    # we calculate parameters to make an initial guess
    n = len(peak_data)                          # the number of data points
    mean = np.sum(peak_data)/n                   
    sigma = (np.sum(clear_data * (peak_data - mean) ** 2)/n) ** 0.5

    # find gaussian fit 
    popt2, pcov2 = curve_fit(gaus, peak_data, clear_data, p0=[np.max(clear_data), mean, sigma], maxfev=90000)
    a, x0, sigma = popt2
    #print(x0, sigma)
    return peak_data,  gaus(peak_data, a, x0, sigma)


#%%
def merge_gaussians(region_array, edges_array):
    '''
    creates array of all background data
    '''

    n_peaks = len(edges_array)
    all_peaks = np.full(n_peaks, None)
    all_masses = np.full(n_peaks, None)

    for i in range(n_peaks):
        all_peaks[i], all_masses[i] = fit_gaussian(region_array[i], edges_array[i])

    # get rid of empty regions 
    all_peaks = np.array(np.concatenate(all_peaks))
    all_peaks = all_peaks[np.where(all_peaks != 0)]

    all_masses = np.array(np.concatenate(all_masses))
    all_masses = all_masses[np.where(all_peaks != 0)]

    return all_peaks, all_masses

#%%

def plot_gaussian():
    '''
    Plots Gaussian fit
    '''
    all_masses, clear_data, exp_fit, x = remove_bg(all_regions, all_edges)      # need to extract all data
    peak_data, gaus_data = merge_gaussians(all_regions, all_edges)

    pylab.plot(peak_data, gaus_data/np.sum(gaus_data), label='best fit gaussians')      # these are normalised to 1 now
    pylab.plot(all_masses, clear_data/np.sum(gaus_data), label='signal data')
    pylab.ylim(0)
    pylab.title('Normalised gaussian fit of data')
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    pylab.legend()
    pylab.show()

plot_gaussian()

#CRYSTALBALLS

def crystal_ball(x, x0, sigma):
    #paper says alpha = 2, n = 1
    A = ((1/np.abs(2))**1)*np.exp((-np.abs(2)**2)/2)
    B = 1/np.abs(2) - np.abs(2)
    C = (1/np.abs(2))*(1/(1.0000001-1))*np.exp(-np.abs(2)**2/2)
    D = np.sqrt(np.pi/2)*(1+erf(np.abs(2)/np.sqrt(2)))
    N = 1/(sigma*(C+D))
    
    sub_a = N*np.exp((-x-x0)**2/2*sigma**2)
    sup_a = N*A*(B-(x-x0)/sigma)**(-1)
    

    for i in len(x):
        if (x[i]-x0)/sigma > -2:
            cb = sub_a
        elif (x[i]-x0)/sigma <= -2:
            cb = sup_a
    
    return cb
    

def fit_crystalBall(region_array, edges_array):
    '''
    use this to loop over peaks and get bg region only
    '''

    all_masses, clear_data, exp_fit, x = remove_bg(all_regions, all_edges)      # need to extract all data

    # need to separate peaks

    _, low_peak_edge, high_peak_edge, _ = edges_array

    # take data outside of background and only within peaks
    peak_data = all_masses[np.where((all_masses >= low_peak_edge) & (all_masses <= high_peak_edge))]
    clear_data = clear_data[np.where((all_masses >= low_peak_edge) & (all_masses <= high_peak_edge))]

    '''
    This fits the peak to a normal distribution
    '''
    # we calculate parameters to make an initial guess
    nopts = len(peak_data)                          # the number of data points
    mean = np.sum(peak_data)/nopts                   
    sigma = (np.sum(clear_data * (peak_data - mean) ** 2)/nopts) ** 0.5

    # find CrystalBall fit 
    popt2, pcov2 = curve_fit(crystal_ball, peak_data, clear_data, p0=[mean, sigma], maxfev=90000)
    x0, sigma = popt2
    #print(x0, sigma)
    return peak_data,  crystal_ball(peak_data, x0, sigma)

def merge_cbs(region_array, edges_array):
    '''
    creates array of all background data
    '''

    n_peaks = len(edges_array)
    all_peaks = np.full(n_peaks, None)
    all_masses = np.full(n_peaks, None)

    for i in range(n_peaks):
        all_peaks[i], all_masses[i] = fit_crystalBall(region_array[i], edges_array[i])

    # get rid of empty regions 
    all_peaks = np.array(np.concatenate(all_peaks))
    all_peaks = all_peaks[np.where(all_peaks != 0)]

    all_masses = np.array(np.concatenate(all_masses))
    all_masses = all_masses[np.where(all_peaks != 0)]

    return all_peaks, all_masses

def plot_crystal():
    '''
    Plots Crystal Ball fit
    '''
    all_masses, clear_data, exp_fit, x = remove_bg(all_regions, all_edges)      # need to extract all data
    peak_data, cb_data = merge_cbs(all_regions, all_edges)

    pylab.plot(peak_data, cb_data/np.sum(cb_data), label='normalised crystal ball fits')      # these are normalised to 1 now
    pylab.plot(all_masses, clear_data/np.sum(cb_data), label='signal data')
    pylab.ylim(0)
    pylab.title('Normalised Crystal Ball fit of data')
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    pylab.legend()
    pylab.show()

plot_crystal()