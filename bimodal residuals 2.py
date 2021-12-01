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


f  =  open("ups-15-small.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.array(np.split(datalist,nevent))

#  make  list  of  invariant  mass  of  events
xmass = xdata[:,0]          # just made this a bit more efficient
xmass_name = r'$Mass ( \mu ^-\mu ^+)$ $(GeV/c^2)$'
xmass_units = r'Candidates/ (25 Mev$/c^2$)'

#%%
region1 =  np.array(xmass)[np.where((xmass > 9.85) & (xmass < 10.18))]


#at this point we are estimating the point where the peak starts to define the background

#here we can define the edges but we'll do that with Niamh's cool method again later
edge1, edge2 = 9.907067045358502, 10.125443712087787

fu = 90
#%%
def plot_histogram(name, values, units, normed=False, num_bins=fu):     
    # find binwidth, use Freeman-Diaconis rule
    
    # plot
    pylab.hist(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed, alpha=0.5, label='background data histogram')
    pylab.ylabel(xmass_units)
    pylab.xlabel(name + " " + units)
    pylab.legend()
    pylab.show()

#%%
def get_bins(values):
    # find binwidth, use Freeman-Diaconis rule
    mass_iqr = stats.iqr(values)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)

    return num_bins


#%%
def background():
    # number of events
    nevents = len(region1)

    # define sideband regions each be half as wide as the signal region 
    side_low = region1[np.where((region1 < edge1) & (region1 > 9.85))]
    side_high = region1[np.where((region1 < 10.18) & (region1 > edge2))]
    empty_regions = np.zeros(len(region1[np.where((region1 >= edge1) & (region1 <= 9.85))]))

    # concatenate these arrays to get a dataset for background
    bg_data = np.hstack((side_low, empty_regions, side_high))

    # render histogram data
    
    bg_counts, bg_masses = np.histogram(bg_data, bins=fu, range=[np.min(bg_data), np.max(bg_data)], density=False)
    bg_masses = bg_masses[:-1] + (bg_masses[1:] - bg_masses[:-1]) / 2

    # get rid of empty regions 
    #bg_masses = bg_masses[np.where(bg_counts != 0)]
    #bg_counts = bg_counts[np.where(bg_counts != 0)]

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

    bg_masses = bg_masses[np.where(bg_counts != 0)]
    bg_counts = bg_counts[np.where(bg_counts != 0)]

    popt, pcov = curve_fit(lambda t, a, b: a * np.exp(-b * t),  bg_masses,  bg_counts, maxfev=90000)


    return popt[0], popt[1]

#%%
def remove_bg():
    #first we need to render histogram data for the whole of region1
    bg_masses, bg_counts, bg_all = background()
    
    all_counts, all_masses = np.histogram(region1, bins=get_bins(bg_masses)-(get_bins(bg_masses)-fu), range=[np.min(region1), np.max(region1)])
    all_masses = all_masses[:-1] + (all_masses[1:] - all_masses[:-1]) / 2

    # create set of points to fit region1 
    x = np.linspace(np.min(all_masses), np.max(all_masses),len(all_masses))
    a, b = fit_bg()
    exp_fit = a * np.exp(-b * x)
    clear_data = all_counts - exp_fit

    return all_masses, clear_data, exp_fit, x

#%%
def plot_bg():
    '''
    We can plot this stuff to visualise our cleared signal
    '''

    all_masses, clear_data, exp_fit, x = remove_bg()
    #ydata = ydata/np.sum(ydata)
    bg_masses, bg_counts, bg_all = background()
    all_counts, _ = np.histogram(region1, bins=get_bins(bg_masses)-(get_bins(bg_masses)-fu), range=[np.min(region1), np.max(region1)])


    pylab.plot(all_masses, clear_data, label='cleared signal')
    pylab.plot(all_masses, all_counts, label='all signals')
    pylab.plot(bg_masses, bg_counts, label='background data')
    pylab.plot(x, exp_fit, label='best fit function')
    pylab.title("Graph of " + str(xmass_name) + " data for the second peak" )
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    pylab.ylim(0)
    pylab.legend()
    plot_histogram(xmass_name, bg_all, xmass_units)

plot_bg()

#%%

def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def bimodal(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return gaus(x, A1, mu1, sigma1) + gaus(x, A2, mu2, sigma2)

def fit_gaussian():
    
    #This fits the first peak to a normal distribution
    
    x, y, ydata = remove_bg()[0:3]

    # we calculate parameters to make an initial guess
    n = len(x)                          # the number of data points
    mean = np.sum(x)/n                   
    sigma = (np.sum(y*(x-mean)**2)/n)**0.5

    # find gaussian fit 
    popt2, pcov2 = curve_fit(bimodal, x, y, p0=[np.max(y), mean, sigma, np.max(y), mean, sigma/10], maxfev=900000)
    A1, mu1, sigma1, A2, mu2, sigma2 = popt2
    
    return x, y, bimodal(x, A1, mu1, sigma1, A2, mu2, sigma2), popt2

#%%
def residuals():
    
    x, y, bimodal, popt2 = fit_gaussian()
    area = np.trapz(bimodal, x)

    pars_1 = popt2[0:3]
    pars_2 = popt2[3:6]
    mean3 = x[np.where(bimodal/area == np.max(bimodal/area))][0]

    gauss_peak_1 = gaus(x, *pars_1)/area
    gauss_peak_2 = gaus(x, *pars_2)/area

    #PLOT
    fig1 = plt.figure(1)
    #Plot Data-model
    frame1 = fig1.add_axes((.1,.3,.8,.6))
    bimodal = bimodal/area
    y = y/area
    plt.title(r'Normalised composite probability curve and residuals for the $\Upsilon(S2) peak$')
    #xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]
    plt.plot(x, gauss_peak_1, 'g', label='first gaussian')
    plt.fill_between(x, gauss_peak_1.min(), gauss_peak_1, facecolor="cyan", alpha=0.5)

    plt.plot(x, gauss_peak_2, 'y', label='second gaussian')
    plt.fill_between(x, gauss_peak_2.min(), gauss_peak_2, facecolor="yellow", alpha=0.5)

    plt.plot(x, y, label='signal data') #Noisy data
    plt.plot(x, bimodal, label='best fit gaussian') #Best fit model
    plt.ylabel('Normalised signal frequency')
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.legend()
    plt.grid(True)
    
    #Residual plot
    difference = bimodal - y
    frame2 = fig1.add_axes((.1,.1,.8,.2))        
    plt.plot(x, difference,'.r')
    plt.vlines(x, difference, np.zeros(len(y)), 'k',alpha=0.3, linewidth=0.6)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.xlabel(xmass_name)
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()


residuals()
#%%
