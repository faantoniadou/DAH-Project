"""
function finding code
"""
import  numpy  as  np
import pylab
from scipy import stats
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import scipy.stats as ss
from numpy import trapz
from scipy.integrate import simps

#%%

f  =  open("ups-15.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.array(np.split(datalist,nevent))

#  make  list  of  invariant  mass  of  events
xmass = xdata[:,0]          # just made this a bit more efficient
xmass_name = r'$Mass ( \mu ^-\mu ^+)$ $(GeV/c^2)$'
xmass_units = r'Candidates/ (25 Mev$/c^2$)'


# here we get peak edges using Niamh's peak finding method
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

fu = 180
#%%

def get_bins(values):
    # find binwidth, use Freeman-Diaconis rule
    mass_iqr = stats.iqr(values)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width) + fu

    return num_bins


def hist_data(values, num_bins, normed=False):
    # renders histogram data
    values, bins = np.histogram(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed)
    
    return values, bins


def hist_area(values, normed=False, num_bins=200):
    #finds area of the histogram
    values, bins = hist_data(values, num_bins, normed)
    area = np.sum(np.diff(bins) * values)

    return area


#%%

def plot_histogram(name, values, units, num_bins, normed=False, label_val='hist'):     
    # plot histogram
    values, bins, _ = pylab.hist(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed, label=label_val, alpha=0.5)

    pylab.title("Histogram of " + name + " data" )
    pylab.ylabel("Counts in bin")
    pylab.xlabel(name + " " + units)


#%%
def bg_data(values, edges_array):
    #use this to loop over peak edges and get bg region only

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
    #creates array of all background data

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
    numbins = (get_bins(all_bg))
    
    bg_counts, bg_masses = hist_data(all_bg, numbins)
    bg_masses = bg_masses[:-1] + (bg_masses[1:] - bg_masses[:-1]) / 2

    return bg_masses, bg_counts, all_bg


#%%
def exp_decay(t, a, b):
    return a * np.exp(-b*t)


#%%
def fit_bg(region_array, edges_array):
    #this fits the background data to an exponential decay function
    #returns the function variables a and b for a * np.exp(-b*t)
    
    bg_masses, bg_counts, all_bg = bg_hist(region_array, edges_array)

    bg_masses = bg_masses[np.where(bg_counts != 0)]
    bg_counts = bg_counts[np.where(bg_counts != 0)]

    popt, pcov = curve_fit(exp_decay,  bg_masses,  bg_counts, p0=[19000, 0.595628], maxfev=900000)

    return popt[0], popt[1]




#%%
def remove_bg(region_array, edges_array):
    #first we need to render histogram data for the whole of region1
    bg_masses, bg_counts, all_bg = bg_hist(region_array, edges_array)

    numbins = get_bins(all_bg)
    
    all_counts, all_masses = hist_data(xmass, numbins)
    all_masses = all_masses[:-1] + (all_masses[1:] - all_masses[:-1]) / 2

    # create set of points to fit  
    x = np.linspace(np.min(xmass), np.max(xmass),len(all_masses))
    a, b = fit_bg(region_array, edges_array)
    clear_data = all_counts - exp_decay(x, a, b)

    return all_masses, clear_data, exp_decay(x, a, b), x

#%%
def plot_bg():
    #We can plot this stuff to visualise our cleared signal

    bg_masses, bg_counts, bg_all = bg_hist(all_regions, all_edges)

    numbins = get_bins(bg_all)

    all_masses, clear_data, exp_fit, x = remove_bg(all_regions, all_edges)

    allcounts, _ = hist_data(xmass, numbins)

    plot_histogram(xmass_name, bg_all[np.where(bg_all != 0)], xmass_units, numbins, normed=False, label_val='background data histogram')
    pylab.plot(all_masses, clear_data, label='cleared signal')
    pylab.plot(all_masses, allcounts, label='all signals')
    pylab.plot(bg_masses, bg_counts, label='background data')
    pylab.plot(x, exp_fit, label='best fit background curve')
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


#%%
def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


#%%
def fit_gaussian(region_array, edges_array):
    #use this to loop over peaks and get bg region only
    
    all_masses, clear_data, exp_fit, x = remove_bg(all_regions, all_edges)      # need to extract all data

    # need to separate peaks

    _, low_peak_edge, high_peak_edge, _ = edges_array

    # take data outside of background and only within peaks
    peak_data = all_masses[np.where((all_masses >= low_peak_edge) & (all_masses <= high_peak_edge))]
    clear_data = clear_data[np.where((all_masses >= low_peak_edge) & (all_masses <= high_peak_edge))]

    #This fits the peak to a normal distribution
    # we calculate parameters to make an initial guess
    n = len(peak_data)                          # the number of data points
    mean = np.sum(peak_data)/n                   
    sigma = (np.sum(clear_data * (peak_data - mean) ** 2)/n) ** 0.5

    # find gaussian fit 
    popt2, pcov2 = curve_fit(gaus, peak_data, clear_data, p0=[np.max(clear_data), mean, sigma/10], maxfev=90000000)
    a, x0, sigma = popt2
    perr = np.sqrt((np.diag(pcov2)))
    gaus_data = gaus(peak_data, popt2[0], popt2[1], popt2[2])
    print(popt2, perr)
    return peak_data, gaus_data, popt2[1], np.max(gaus_data)


#%%
def merge_gaussians(region_array, edges_array):
    #creates array of all background data
    
    n_peaks = len(edges_array)
    all_peaks = np.full(n_peaks, None)
    all_masses = np.full(n_peaks, None)
    means = np.full(n_peaks,None)
    maxima = np.full(n_peaks,None)

    for i in range(n_peaks):
        all_peaks[i], all_masses[i] = fit_gaussian(region_array[i], edges_array[i])[0:2]
        means[i] =  fit_gaussian(region_array[i], edges_array[i])[2]
        maxima[i] = fit_gaussian(region_array[i], edges_array[i])[-1]

    # get rid of empty regions 
    all_peaks = np.array(np.concatenate(all_peaks))

    all_masses = np.array(np.concatenate(all_masses))


    return all_peaks, all_masses, means, maxima


#%%

def plot_gaussian():
    #Plots Gaussian fit

    all_masses, clear_data, exp_fit, x = remove_bg(all_regions, all_edges)      # need to extract all data
    peak_data, gaus_data, means, maxima = merge_gaussians(all_regions, all_edges)
    area = np.trapz(gaus_data, peak_data)

    maxima = maxima/np.sum(gaus_data)
    pylab.plot(peak_data, gaus_data/np.sum(gaus_data), label='best fit gaussians')      # these are normalised to 1 now
    pylab.plot(all_masses, clear_data/np.sum(gaus_data), label='signal data')

    pylab.vlines(x = means[0], ymin = 0, ymax = maxima[0], ls='--', lw=0.75, colors = 'blue', label = 'peak 1 : mean = ' + str(np.round(means[0],4)))
    pylab.vlines(x = means[1], ymin = 0, ymax = maxima[1], ls='--', lw=0.75, colors = 'red', label = 'peak 2 : mean = ' + str(np.round(means[1],4)))
    pylab.vlines(x = means[2], ymin = 0, ymax = maxima[2], ls='--', lw=0.75, colors = 'green', label = 'peak 3 : mean = ' + str(np.round(means[2],4)))
    

    pylab.ylim(0)
    pylab.title('Normalised Gaussian fit of mass spectrum')
    pylab.xlabel(xmass_name)
    pylab.ylabel('Normalised signal frequency')
    pylab.legend()
    pylab.show()

plot_gaussian()

# %%

def residuals():
    
    all_masses, clear_data, exp_fit, x = remove_bg(all_regions, all_edges)
    peak_data, gaus_data, means, maxima = merge_gaussians(all_regions, all_edges)
    area = np.trapz(gaus_data)

    dummy = [x if x in peak_data else 0 for x in all_masses]
    dummy = np.array(dummy)
    clear_data = clear_data[dummy != 0]


    #PLOT
    fig1 = plt.figure(1)
    #Plot Data-model
    frame1 = fig1.add_axes((.1,.3,.8,.6))
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]

    clear_data = clear_data[np.where(gaus_data/area != 0)]

    maxima = maxima/area
    pylab.plot(peak_data, gaus_data/area, 'm', label='best fit gaussians')      # these are normalised to 1 now
    pylab.plot(peak_data, clear_data/area,'k-',linewidth=0.8, label='signal data')

    pylab.vlines(x = means[0], ymin = 0, ymax = maxima[0], ls='--', lw=0.75, colors = 'blue', label = 'peak 1 : mean = ' + str(round(means[0],4)))
    pylab.vlines(x = means[1], ymin = 0, ymax = maxima[1], ls='--', lw=0.75, colors = 'red', label = 'peak 2 : mean = ' + str(round(means[1],4)))
    pylab.vlines(x = means[2], ymin = 0, ymax = maxima[2], ls='--', lw=0.75, colors = 'green', label = 'peak 3 : mean = ' + str(round(means[2],4)))
    
    pylab.title('Normalised gaussian fit of data')
    pylab.xlabel(xmass_name)
    pylab.ylabel('Normalised signal frequency')
    pylab.legend()
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.grid(True)

    
    #Residual plot
    difference = gaus_data/area - clear_data/area
    frame2 = fig1.add_axes((.1,.1,.8,.2))        
    pylab.plot(peak_data, difference,'.r')
    pylab.vlines(peak_data, difference, np.zeros(len(difference)), 'k',alpha=0.3, linewidth=0.6)
    pylab.axhline(y=0, color='k', linestyle='-')
    pylab.xlabel(xmass_name)
    pylab.ylabel('Residuals')
    pylab.grid(True)
    pylab.show()

    return None


residuals()
# %%
