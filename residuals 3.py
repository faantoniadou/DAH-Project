"""
function finding code
"""

import  numpy as np
import pylab
from scipy import stats
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import scipy.stats as ss


f  =  open("ups-15.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.array(np.split(datalist,nevent))

#  make  list  of  invariant  mass  of  events
xmass = xdata[:,0]          # just made this a bit more efficient
xmass_name = r'$Mass ( \mu ^-\mu ^+)$ $(GeV/c^2)$'
xmass_units = r'Candidates/ (25 Mev$/c^2$)'

#%%
region1 =  np.array(xmass)[np.where((xmass > 10.19) & (xmass < 10.55))]


#at this point we are estimating the point where the peak starts to define the background

#here we can define the edges but we'll do that with Niamh's cool method again later
edge1, edge2 = 10.251361855283378, 10.464073172792794
mass_iqr = stats.iqr(region1)
bin_width = 2 * mass_iqr/((len(region1))**(1/3))    
num_bins = int(2/bin_width) - 400

#%%
def plot_histogram(name, values, units, normed=False):     
    # find binwidth, use Freeman-Diaconis rule
    
    # plot
    pylab.hist(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed, alpha=0.5, label='background data histogram')
    pylab.title("Histogram of " + name + " data" )
    pylab.ylabel(xmass_units)
    pylab.xlabel(name + " " + units)
    pylab.legend()
    pylab.show()




#%%
def background():
    # number of events
    nevents = len(region1)

    # define sideband regions each be half as wide as the signal region 
    side_low = xmass[np.where((xmass < edge1) & (xmass > 10.19))]
    side_high = xmass[np.where((xmass < 10.55) & (xmass > edge2))]
    empty_regions = np.zeros(len(xmass[np.where((xmass > edge1) & (xmass <= 9.0))]))

    # concatenate these arrays to get a dataset for background
    bg_data = np.hstack((side_low, empty_regions, side_high))

    # render histogram data
    
    bg_counts, bg_masses = np.histogram(bg_data, bins=num_bins, range=[np.min(bg_data), np.max(bg_data)], density=False)

    # get rid of empty regions 
    #bg_masses = bg_masses[np.where(bg_counts != 0)]
    #bg_counts = bg_counts[np.where(bg_counts != 0)]

    return bg_masses[0:-1], bg_counts, bg_data


#%%
def fit_bg():
    '''
    this fits the background data to an exponential decay function
    returns the function variables a and b for a * np.exp(-b*t)
    '''
    bg_masses = background()[0]
    bg_all = background()[2]
    bg_counts = background()[1] 

    bg_masses = bg_masses[np.where(bg_counts != 0)]
    bg_counts = bg_counts[np.where(bg_counts != 0)]

    popt, pcov = curve_fit(lambda t, a, b: a * np.exp(-b * t),  bg_masses,  bg_counts, maxfev=90000)

    # random dataset to display exponential decay function 
    x = np.linspace(np.min(bg_masses), np.max(bg_masses),1000)

    return popt[0], popt[1]

#%%
def remove_bg():
    #first we need to render histogram data for the whole of region1
    bg_masses, bg_counts, bg_all = background()
    all_counts, all_masses = np.histogram(region1, bins=num_bins, range=[np.min(region1), np.max(region1)])
    
    # create set of points to fit region1 
    x = np.linspace(np.min(all_masses), np.max(all_masses),len(all_masses)-1)
    a, b = fit_bg()
    exp_fit = a * np.exp(-b * x)
    clear_data = all_counts - exp_fit

    return all_masses[0:-1], clear_data, exp_fit, x


#%%

def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fit_gaussian():
    
    #This fits the first peak to a normal distribution
    
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
def residuals():
    
    x, y, a, x0, sigma, gauss = fit_gaussian()      # need to extract all data

    #PLOT
    fig1 = plt.figure(1)
    #Plot Data-model
    frame1 = fig1.add_axes((.1,.3,.8,.6))
    area = np.trapz(gauss,x)

    gauss = gauss/area
    y = y/area
    plt.title(r'Best fit Gaussian and residuals for the $\Upsilon(3S) peak$')
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.plot(x, y, label='signal data') #Noisy data
    plt.plot(x, gauss, label='best fit gaussian') #Best fit model
    plt.ylabel('Normalised signal frequency')
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.legend()
    plt.grid(True)
    
    #Residual plot
    difference = gauss - y
    frame2 = fig1.add_axes((.1,.1,.8,.2))        
    plt.plot(x, difference,'.r')
    plt.vlines(x, difference, np.zeros(len(y)), 'k',alpha=0.3, linewidth=0.6)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.xlabel(xmass_name)
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()


residuals()


# %%
