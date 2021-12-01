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
import scipy.integrate as integrate
from scipy.integrate import quad


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


#at this point we are estimating the point where the peak starts to define the background

#here we can define the edges but we'll do that with Niamh's cool method again later

edge1 = 9.29
edge2 = 9.61


#%%
def plot_histogram(name, values, units, normed=False):     
    # find binwidth, use Freeman-Diaconis rule
    mass_iqr = stats.iqr(region1)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)

    # plot
    pylab.hist(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed, alpha=0.5, label='background data histogram')
    pylab.title("Histogram of " + name + " data" )
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


    mass_iqr = stats.iqr(region1)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)
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
def plot_gaussian():

    #Plots Gaussian fit

    x, y, bimodal, popt2 = fit_gaussian()
    pars_1 = popt2[0:3]
    pars_2 = popt2[3:6]
    area = np.trapz(bimodal, x)

    mean3 = x[np.where(bimodal/area == np.max(bimodal/area))][0]
    print(mean3)

    gauss_peak_1 = gaus(x, *pars_1)/area
    gauss_peak_2 = gaus(x, *pars_2)/area

    pylab.plot(x, gauss_peak_1, 'c')#, label='first gaussian')
    pylab.fill_between(x, gauss_peak_1.min(), gauss_peak_1, facecolor="cyan", alpha=0.5)

    pylab.plot(x, gauss_peak_2, 'y')#, label='second gaussian')
    pylab.fill_between(x, gauss_peak_2.min(), gauss_peak_2, facecolor="yellow", alpha=0.5)    
    plt.vlines(x = pars_1[1], ymin = 0, ymax = max(gauss_peak_1), ls='--', lw=0.75, colors = 'purple', label = 'gaussian 1 mean = '+str(round(pars_1[1],4)))
    plt.vlines(x = pars_2[1], ymin = 0, ymax = max(gauss_peak_2), ls='--', lw=0.75, colors = 'magenta', label = 'gaussian 2 mean = ' + str(round(pars_2[1],4)))
    plt.vlines(x = mean3, ymin = 0, ymax = np.max(bimodal/area), ls='--', lw=0.75, colors = 'red', label = 'composite curve mean = ' + str(round(mean3,4)))

    pylab.plot(x, bimodal/area, 'm', label='best fit overlapping gaussians')      # these are normalised to 1 now
    
    pylab.plot(x, y/area, 'k', linewidth=0.8, label='signal data')
    pylab.ylim(0)
    pylab.title(r'Normalised composite probability curve for the $\Upsilon$(S1) peak')
    pylab.xlabel(xmass_name)
    pylab.xticks(np.arange(9, 9.80, 0.15))
    pylab.ylabel(xmass_units)
    pylab.legend()
    pylab.show()

plot_gaussian()


#%%
def plot_composite():
    #Plots composite probability curve
    x1, y1, bimodal, popt2 = fit_gaussian()
    pars_1 = popt2[0:3]
    pars_2 = popt2[3:6]

    mass_iqr = stats.iqr(region1)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)

    all_counts, all_masses = np.histogram(region1, bins=num_bins, range=[np.min(region1), np.max(region1)])
    x2, x3, y2 = remove_bg()[0:3]

    #y_composite = y_composite*2
    #y_composite = y_composite/2333.33333
    # normalise the composite probability
    
    gauss_peak_1 = gaus(x1, *pars_1)
    gauss_peak_2 = gaus(x1, *pars_2)

    y_composite = gauss_peak_1+y2+gauss_peak_2
    area = np.trapz(y_composite, x1)         # calculate area to normalise graph

    mean3 = x1[np.where(y_composite/area == np.max(y_composite/area))]     


    gauss_peak_1 = gauss_peak_1/area
    gauss_peak_2 = gauss_peak_2/area

    pylab.plot(x1, all_counts/area,'r', label='signal data')
    pylab.plot(x1, gauss_peak_1, 'b')#, label='first gaussian')
    pylab.fill_between(x1, gauss_peak_1.min(), gauss_peak_1, facecolor="blue", alpha=0.5)

    pylab.plot(x1, gauss_peak_2, 'c')#, label='second gaussian')
    pylab.fill_between(x1, gauss_peak_2.min(), gauss_peak_2, facecolor="cyan", alpha=0.5)

    pylab.plot(x1, y2/area, 'y', label='background fit')
    #pylab.plot(x2, y_composite, label='best fit probability curve')
    pylab.fill_between(x1, 0, y2/area, facecolor="yellow", alpha=0.5)
    

    plt.vlines(x = pars_1[1], ymin = 0, ymax = max(gauss_peak_1), ls='--', lw=0.75, colors = 'purple', label = 'gaussian 1 mean = '+str(round(pars_1[1],4)))
    plt.vlines(x = pars_2[1], ymin = 0, ymax = max(gauss_peak_2), ls='--', lw=0.75, colors = 'magenta', label = 'guassian 2 mean = ' + str(round(pars_2[1],4)))
    plt.vlines(x = mean3, ymin = 0, ymax = np.max(y_composite/area), ls=':', lw=0.75, colors = 'black', label = 'composite curve mean = ' + str(round(mean3[0], 4)) )


    pylab.plot(x2, y_composite/area, 'k', label='best fit composite PDF')
    pylab.xlim(np.min(x1),np.max(x1))
    pylab.xlabel(xmass_name)
    pylab.ylabel(xmass_units)
    pylab.xticks(np.arange(9, 9.80, 0.15))
    pylab.legend()
    plt.ylim(0)
    pylab.title(r'Normalised composite probability curve for the $\Upsilon$(S1) peak')
    #plot_histogram(xmass_name, region1, xmass_units, True)
    
    pylab.show()

plot_composite()
