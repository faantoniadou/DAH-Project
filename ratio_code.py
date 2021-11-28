# -*- coding: utf-8 -*-
#,8.399752474068893e-05
#0.0001278535384062904
"""
Created on Sun Nov 28 15:23:55 2021

@author: clark
"""
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import stats as ss
import pylab
from scipy.optimize import curve_fit


f  =  open("ups-15.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.array(np.split(datalist,nevent))


"""
Following 3 lines are used to perform slices on data, can use to either cut out 
data to be ignored or to specify ranges for specific analyses eg production ratios
"""
#next line defined s.t. analysis carried out only for pT of dimuon pair in range 8-10 GeV
#change limits here to analyse for different pT ranges.
xdata = np.array(xdata)[np.where((xdata[:,1] <= 10)& (xdata[:,1] >=8) )]
#xdata = np.array(xdata)[np.where((xdata[:,2] >= 2)&(xdata[:,2]<=6))]
#  make  list  of  invariant  mass  of  events
xmass = xdata[:,0]          
xmass_name = r'$Mass ( \mu ^-\mu ^+)$ $(GeV/c^2)$'
xmass_units = r'Candidates/ (25 Mev$/c^2$)'



edge1 = 9.28981606785984
edge2 = 9.604490115001488
edge3 = 9.907067045358502
edge4 = 10.125443712087787
edge5 = 10.251361855283378
edge6 = 10.464073172792794

low_side_edge1, low_peak_edge1, high_peak_edge1, high_side_edge1 = [9.0, 9.22981606785984, 9.654490115001488, 9.77]
low_side_edge2, low_peak_edge2, high_peak_edge2, high_side_edge2 = [9.77, 9.837067045358502, 10.195243712087787, 10.1953]
low_side_edge3, low_peak_edge3, high_peak_edge3, high_side_edge3 = [10.1953, 10.195361855283378, 10.494073172792794, 11]

edges1 = np.array((low_side_edge1, low_peak_edge1, high_peak_edge1, high_side_edge1))
edges2 = np.array((low_side_edge2, low_peak_edge2, high_peak_edge2, high_side_edge2))
edges3 = np.array((low_side_edge3, low_peak_edge3, high_peak_edge3, high_side_edge3))

# define each peak region
region1 = np.array(xmass)[np.where((xmass > low_side_edge1) & (xmass < high_side_edge1))]
region2 = np.array(xmass)[np.where((xmass > low_side_edge2) & (xmass < high_side_edge2))]
region3 = np.array(xmass)[np.where((xmass > low_side_edge3) & (xmass < high_side_edge3))]

all_edges = np.array((edges1, edges2, edges3), dtype=object)
all_regions = np.array([region1, region2, region3], dtype=object)

def get_bins(values):
    # find binwidth, use Freeman-Diaconis rule
    mass_iqr = ss.iqr(values)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)

    return num_bins

def hist_data(values, num_bins, normed=False):
    # renders histogram data
    values, bins = np.histogram(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed)

    return values, bins


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



def fit_bg(region_array, edges_array):
    '''
    this fits the background data to an exponential decay function
    returns the function variables a and b for a * np.exp(-b*t)
    '''

    bg_masses, bg_counts, all_bg = bg_hist(region_array, edges_array)

    bg_masses = bg_masses[np.where(bg_counts != 0)]
    bg_counts = bg_counts[np.where(bg_counts != 0)]

    popt, pcov = curve_fit(exp_decay,  bg_masses,  bg_counts, p0=[19000, 0.595628], maxfev=9000000)

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


#######################################################################################################
#######################################################################################################
#######################################################################################################
###################################### GAUSSIAN FITS NOW ##############################################
#######################################################################################################
#######################################################################################################
#######################################################################################################


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
    print(x0, sigma)
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
    pylab.plot(peak_data, gaus_data)
    pylab.show()
    
   #this code is for the ratios
    sum_over_range = np.sum(clear_data)
    n1count = np.sum(clear_data[np.where((all_masses>(9.445-(2*0.045403628544171515)))&(all_masses<(9.445171412156942+(2*0.045403628544171515))))])
    n2count = np.sum(clear_data[np.where((all_masses>(10.008-(2*0.053594361870889604)))&(all_masses<(10.008168666928341+(2*0.053594361870889604))))])
    n3count = np.sum(clear_data[np.where((all_masses>(10.343-(2*0.05896508983331981)))&(all_masses<(10.343+(2*0.05896508983331981))))])
    n2_n1 = n2count/n1count
    n3_n1 = n3count/n1count
    delta_n2_n1 = np.sqrt((((n2_n1)**2)*((np.sqrt(n1count)/n1count**2)*(0.005*n1count)**2*(np.sqrt(n2count)/n2count)*(0.005*n2count)**2)))
    delta_n3_n1 = np.sqrt((((n3_n1)**2)*((np.sqrt(n1count)/n1count**2)*(0.005*n1count)**2*(np.sqrt(n3count)/n3count))*(0.005*n3count)**2))
    print("2-4 GeV")
    print(n1count, np.sqrt(n1count))
    print(n2_n1, delta_n2_n1)
    print(n3_n1, delta_n3_n1)
    
plot_gaussian()

#midpoints of ranges over which pT is sliced 
pT = (1,3,5,7,9)
#ratio N(S2)/N(S1) (number of each in given pT range) and errors
n2_n1 = (0.2369894068096551, 0.24676942525522258, 0.260418277421809, 0.29008173008354476, 0.3167158550021635)
delta_n2_n1 = (7.407936083487728E-05, 4.763755303248949e-05, 5.660411224335539e-05, 8.200353319144654e-05, 8.399752474068893e-05)
#ratio N(S3)/N(S1) (ratio of number produced in given pT range) and errors
n3_n1 = (0.12341806390912437,0.13284935445783208, 0.1453851109305531, 0.16676013556635055, 0.18088731832609958)
delta_n3_n1 = (4.54134580633532e-05, 2.9939911990855904e-05, 3.6558160586191566e-05, 5.4139153579428896e-05, 0.0001278535384062904)

plt.scatter(pT, n2_n1, c = 'black',marker = "v", s = 14, label = 'N(S2)/N(S1)')
plt.scatter(pT, n3_n1, c = 'blue', marker = "^", s = 14, label = 'N(S3)/N(S1)')

plt.errorbar(pT, n3_n1, xerr = 0.5, yerr = delta_n3_n1, capsize = 0, ls='none', color= 'blue')

plt.legend()
plt.errorbar(pT, n2_n1, xerr = 0.5, yerr = delta_n2_n1, capsize = 0, ls='none', color= 'black')
plt.xticks(np.arange(0, 11))
plt.xlabel("Dimuon $p_T$ bins $[GeV/c]$")
plt.ylabel("Ratio N(Sx)/N(S1)")
plt.title("Ratios N(S2)/N(S1) and N(S3)/N(S1) produced, binned by $p_T$")
plt.show()
