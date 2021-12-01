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


f  =  open("ups-15.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.array(np.split(datalist,nevent))

#  make  list  of  invariant  mass  of  events
xmass = xdata[:,0]
xmass_name = r'$Mass ( \mu ^-\mu ^+)$ $(GeV/c^2)$'
xmass_units = r'Candidates/ (25 Mev$/c^2$)'

#%%
region1 =  np.array(xmass)[np.where((xmass > 9.0) & (xmass < 9.75))]
mass_iqr = stats.iqr(region1)
bin_width = 2 * mass_iqr/((nevent)**(1/3))    
num_bins = int(2/bin_width)
#at this point we are estimating the point where the peak starts to define the background

#here we can define the edges but we'll do that with Niamh's cool method again later

edge1, edge2 = 9.28981606785984, 9.604490115001488


#%%
def plot_histogram(name, values, units, normed=False):     
    # find binwidth, use Freeman-Diaconis rule
    mass_iqr = stats.iqr(region1)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)

    # plot
    pylab.hist(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed, alpha=0.5, label='background data histogram',color='green')
    #pylab.title("Histogram of " + name + " data" )
    pylab.ylabel(xmass_units)
    pylab.xlabel(name + " " + units)
    pylab.legend()
    pylab.show()





def background():
    # number of events
    nevents = len(region1)

    # define sideband regions each be half as wide as the signal region 
    side_low = xmass[np.where((xmass < edge1) & (xmass > 9.0))]
    side_high = xmass[np.where((xmass < 9.75) & (xmass > edge2))]
    empty_regions = np.zeros(len(xmass[np.where((xmass > edge1) & (xmass <= 9.0))]))

    # concatenate these arrays to get a dataset for background
    bg_data = np.hstack((side_low, empty_regions, side_high))

    # render histogram data
    mass_iqr = stats.iqr(region1)           # here we use region1 to maintain consistency and avoid different sized bins being used
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)
    bg_counts, bg_masses = np.histogram(bg_data, bins=num_bins, range=[np.min(bg_data), np.max(bg_data)], density=False)

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


    mass_iqr = stats.iqr(region1)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)
    all_counts, all_masses = np.histogram(region1, bins=num_bins, range=[np.min(region1), np.max(region1)])
    
    # create set of points to fit region1 
    x = np.linspace(np.min(all_masses), np.max(all_masses),len(all_masses)-1)
    a, b = fit_bg()
    exp_fit = a * np.exp(-b * x)
    clear_data = all_counts - exp_fit

    all_masses = all_masses[:-1] + (all_masses[1:] - all_masses[:-1]) / 2

    return all_masses, clear_data, exp_fit, x


#%%

def plot_bg():

    #We can plot this stuff to visualise our cleared signal

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
    pylab.title("Graph of " + str(xmass_name) + " data for the first peak" )
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    plt.xticks(np.arange(9, 9.80, 0.15))
    pylab.ylim(0)
    pylab.legend()
    plot_histogram(xmass_name, bg_all, xmass_units)

plot_bg()

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
    perr = np.sqrt(np.diag(pcov2))
    print(popt2,perr)

    return x, y, a, x0, sigma, gaus(x, a, x0, sigma)

#%%
def plot_gaussian():

    #Plots Gaussian fit

    x, y, a, x0, sigma, gauss = fit_gaussian()
    area = np.trapz(gauss, x)

    pylab.plot(x, gauss/area, 'm', label='best fit gaussian')
    pylab.plot(x, y/area,'k', label='signal data')
    pylab.ylim(0)
    pylab.title(r'Normalised gaussian fit of data for the $\Upsilon(S1)$ peak')
    pylab.xlabel(xmass_name)
    plt.xticks(np.arange(9, 9.80, 0.15))
    pylab.ylabel(xmass_units)
    pylab.legend()
    pylab.show()

plot_gaussian()



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

    plt.title(r'Best fit Gaussian and residuals for the $\Upsilon(S1)$ peak ')
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.plot(x, y, 'k-',linewidth=0.8,label='signal data') #Noisy data
    plt.plot(x, gauss,'m', label='best fit gaussian') #Best fit model
    plt.ylabel('Normalised signal frequency')
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.vlines(x = x[np.where(gauss==np.max(gauss))], ymin = 0, ymax = max(gauss), ls='--', lw=0.75, colors = 'blue', label = 'mean = '+str(np.round_(x[np.where(gauss==np.max(gauss))][0], 4)))
    plt.legend()
    plt.ylim(-0.001)
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

    #return difference


residuals()

#%%


def plot_composite():
    #plots composite probability curve
    x, y, a, x0, sigma, gauss = fit_gaussian()      # need to extract all data
    x1, y1 = fit_gaussian()[0], fit_gaussian()[5]
    x2,x3,y2 = remove_bg()[0:3]
    y_composite = y1+y2
    area = np.trapz(y_composite, x1)

    y_composite = y_composite/area    # normalise the composite probability
    # the 750 factor above is so that the area is 1 and fits to the histogram
    pylab.plot(x1,y_composite, label='best fit probability curve')
    pylab.xlim(np.min(x1),np.max(x1))
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    plt.xticks(np.arange(9, 9.80, 0.15))
    pylab.title(r'Normalised composite probability curve for the $\Upsilon(S1)$ peak')
    plot_histogram(xmass_name, region1, xmass_units, True)
plot_composite()


# %%
