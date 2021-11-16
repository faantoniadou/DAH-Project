"""
function finding code
"""
#%%
import  numpy  as  np
import pylab
from scipy import stats
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import scipy.stats as ss
from numpy import trapz
from scipy.integrate import simps


f  =  open("ups-15-small.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.array(np.split(datalist,nevent))

#  make  list  of  invariant  mass  of  events
xmass = xdata[:,0]          # just made this a bit more efficient
xmass_name = r'$Mass ( \mu ^-\mu ^+)$ $(GeV/c^2)$'
xmass_units = r'Candidates/ (25 Mev$/c^2$)'


'''
at this point we are estimating the point where the peak starts to define the background

here we can define the edges but we'll do that with Niamh's cool method again later
'''
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
'''
I broke this down because it makes normalising a bit easier
'''

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


def hist_area(values, normed=False, num_bins=200):
    #finds area of the histogram
    values, bins = hist_data(values, num_bins, normed)
    area = np.sum(np.diff(bins) * values)

    return area


#%%

def plot_histogram(name, values, units, num_bins, normed=False):     
        # plot
    values, bins, _ = pylab.hist(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed, label='histogram')

    pylab.title("Histogram of " + name + " data" )
    pylab.ylabel("Counts in bin")
    pylab.xlabel(name + " " + units)


plot_histogram(xmass_name, xmass, xmass_name, get_bins(xmass), False)


#%%
def bg_data(values, edges_array):
    '''
    use this to loop over peak edges and get bg region only
    '''

    low_side_edge, low_peak_edge, high_peak_edge, high_side_edge = edges_array.tolist()
    
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


#print(len(np.concatenate( all_regions, axis=0 )))
#print(len(all_bg))




#%%
def bg_hist(region_array, edges_array):
    # render histogram data
    all_bg = bg_merge_data(region_array, edges_array)
    num_bins = (get_bins(xmass)) 
    
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

    return popt[0], popt[1]

print(fit_bg(all_regions, all_edges))


#%%
def remove_bg(region_array, edges_array):
    #first we need to render histogram data for the whole of region1
    bg_masses, bg_counts, all_bg = bg_hist(region_array, edges_array)

    numbins = get_bins(xmass)
    
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
    numbins1 = get_bins(xmass)

    teras, clear_data, exp_fit, x = remove_bg(all_regions, all_edges)

    allcounts, all_masses = hist_data(xmass, numbins1)

    bg_masses, bg_counts, bg_all = bg_hist(all_regions, all_edges)

    all_masses = all_masses[0:-1]
    plot_histogram(xmass_name, bg_all[np.where(bg_all != 0)], xmass_units, numbins1, normed=False)
    pylab.plot(all_masses, clear_data, label='cleared signal')
    pylab.plot(all_masses, allcounts, label='all signals')
    pylab.plot(bg_masses, bg_counts, label='background data')
    pylab.plot(x, exp_fit, label='best fit function')
    pylab.title("Graph of " + str(xmass_name) + " data" )
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    pylab.xlim(np.min(xmass), np.max(xmass))
    pylab.ylim(0)
    pylab.legend()
    pylab.show()

plot_bg()


def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

#%%
def fit_gaussian():
    '''
    This fits the first peak to a normal distribution
    '''
    x, y, ydata = remove_bg()[0:3]

    # we calculate parameters to make an initial guess
    n = len(x)                          # the number of data points
    mean = np.sum(x)/n                   
    sigma = (np.sum(y*(x-mean)**2)/n)**0.5

    # find gaussian fit 
    popt2, pcov2 = curve_fit(gaus, x, y, p0=[np.max(y), mean, sigma], maxfev=900000)
    a, x0, sigma = popt2

    return x, y, a, x0, sigma, gaus(x, a, x0, sigma)

#%%
def plot_gaussian():
    '''
    Plots Gaussian fit
    '''
    x, y, a, x0, sigma, gauss = fit_gaussian()
    pylab.plot(x, gauss/np.sum(gauss), label='best fit gaussian')      # these are normalised to 1 now
    pylab.plot(x, y/np.sum(y), label='signal data')
    pylab.ylim(0)
    pylab.title('Normalised gaussian fit of data')
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    pylab.legend()
    pylab.show()

plot_gaussian()

# %%
def plot_composite():
    '''
    Plots composite probability curve
    '''

    area_hist = hist_area(region1, False)

    x1, y1 = fit_gaussian()[0], fit_gaussian()[5]
    x2,x3,y2 = remove_bg()[0:3]
    y_composite = y1 + y2

    area = simps(y_composite, dx=np.max(region1) - np.min(region1))

    print('Area under graph = ' + str(area))
    print('Area of histogram = ' + str(area_hist))

    y_composite = y_composite * np.sum(y_composite)*2/area_hist*10**(-6)    # normalise the composite probability
    plot_histogram(xmass_name, region1, xmass_units, normed=True) 

    pylab.plot(x1, y_composite, label='composite probability')
    pylab.xlim(np.min(x1),np.max(x1))
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    pylab.title('Normalised composite probability curve')
    pylab.legend()

    pylab.show

plot_composite()

# %%
