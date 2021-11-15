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

region1 =  np.array(xmass)[np.where((xmass > 9.0) & (xmass < 9.75))]

'''
at this point we are estimating the point where the peak starts to define the background

here we can define the edges but we'll do that with Niamh's cool method again later
'''
edge1 = 9.29
edge2 = 9.61


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


def hist_data(values, normed=False):
    # renders histogram data
    num_bins = get_bins(values)
    values, bins = np.histogram(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed)
    
    return values, bins


def hist_area(values, normed):
    #finds area of the histogram
    values, bins = hist_data(values, normed)
    area = np.sum(np.diff(bins) * values)

    return area


def plot_histogram(name, values, units, normed=False):     
    
    num_bins = get_bins(values)
    # plot
    values, bins, _ = pylab.hist(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed)

    pylab.title("Histogram of " + name + " data" )
    pylab.ylabel("Counts in bin")
    pylab.xlabel(name + " " + units)
    #pylab.show()


plot_histogram(xmass_name, region1, xmass_units, normed=True)


#%%
def background():
    # number of events
    nevents = len(region1)

    # define sideband regions each be half as wide as the signal region 
    side_low = xmass[np.where((xmass < edge1) & (xmass > 9.0))]
    side_high = xmass[np.where((xmass < 9.75) & (xmass > edge2))]

    # concatenate these arrays to get a dataset for background
    bg_data = np.hstack((side_low, side_high))

    # render histogram data
    mass_iqr = stats.iqr(region1)           # here we use region1 to maintain consistency and avoid different sized bins being used
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)
    bg_counts, bg_masses = np.histogram(bg_data, bins=num_bins, range=[np.min(bg_data), np.max(bg_data)], density=False)

    # get rid of empty regions 
    bg_masses = bg_masses[np.where(bg_counts != 0)]
    bg_counts = bg_counts[np.where(bg_counts != 0)]

    return bg_masses, bg_counts, bg_data


#%%
def fit_bg():
    '''
    this fits the background data to an exponential decay function
    returns the function variables a and b for a * np.exp(-b*t)
    '''
    bg_masses = background()[0]
    bg_all = background()[2]
    bg_counts = background()[1] 

    popt, pcov = curve_fit(lambda t, a, b: a * np.exp(-b*t),  bg_masses,  bg_counts, maxfev=90000)

    # random dataset to display exponential decay function 
    x = np.linspace(np.min(bg_masses), np.max(bg_masses),1000)

    return popt[0], popt[1]

#%%
def remove_bg():
    #first we need to render histogram data for the whole of region1
    bg_masses, bg_counts, bg_all = background()
   
    num_bins = get_bins(region1)
    all_counts, all_masses = np.histogram(region1, bins=num_bins, range=[np.min(region1), np.max(region1)])
    
    # create set of points to fit region1 
    x = np.linspace(np.min(all_masses), np.max(all_masses),len(all_masses)-1)
    a, b = fit_bg()
    exp_fit = a * np.exp(-b * x)
    clear_data = all_counts - exp_fit

    return all_masses[0:-1], clear_data, exp_fit, x


#%%
def plot_bg():
    '''
    We can plot this stuff to visualise our cleared signal
    '''
    mass_iqr = stats.iqr(region1)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)

    all_masses, clear_data, exp_fit, x = remove_bg()
    #ydata = ydata/np.sum(ydata)
    all_counts, all_masses = np.histogram(region1, bins=num_bins, range=[np.min(region1), np.max(region1)])
    bg_masses, bg_counts, bg_all = background()

    pylab.plot(all_masses[0:-1], clear_data, label='cleared signal')
    pylab.plot(all_masses[0:-1], all_counts, label='all signals')
    pylab.plot(bg_masses, bg_counts, label='background data')
    pylab.plot(x, exp_fit, label='best fit function')
    pylab.title("Graph of " + str(xmass_name) + " data" )
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    pylab.xlim(np.min(region1), np.max(region1))
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

    pylab.plot(x1, y_composite)
    pylab.xlim(np.min(x1),np.max(x1))
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    pylab.title('Normalised composite probability curve')
    
    pylab.show

plot_composite()

# %%