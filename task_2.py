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


def plot_histogram(name, values, units, normed=False):     
    # find binwidth, use Freeman-Diaconis rule
    mass_iqr = stats.iqr(values)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)

    # plot
    pylab.hist(values,  bins=num_bins,  range=[np.min(values), np.max(values)], density=normed)
    pylab.title("Histogram of " + name + " data" )
    pylab.ylabel("Counts in bin")
    pylab.xlabel(name + " " + units)
    pylab.show()

f  =  open("ups-15-small.bin","rb")
datalist  =  np.fromfile(f, dtype=np.float32)
#  number  of  events
nevent  =  int(len(datalist)/6)
xdata  =  np.split(datalist,nevent)

#  make  list  of  invariant  mass  of  events
xmass  =  []
xmass_name = str("Mass")
xmass_units = str("[GeV/c^2]")
for  i  in  range(0,nevent):
    xmass.append(xdata[i][0])

#plot_histogram(xmass_name, xmass, xmass_units)
xmass = np.array(xmass)

region1 =  np.array(xmass)[np.where((xmass > 9.0) & (xmass < 9.75))]
#plot_histogram(xmass_name, region1, xmass_units)

'''
at this point we are estimating the point where the peak starts to define the background

here we can define the edges but we'll do that with Niamh's cool method again later
'''
edge1 = 9.29
edge2 = 9.61


def background():
    # number of events
    nevents = len(region1)
    #print(N_X)

    # define sideband regions each be half as wide as the signal region 
    side_low = xmass[np.where((xmass < edge1) & (xmass > 9.0))]
    side_high = xmass[np.where((xmass < 9.75) & (xmass > edge2))]

    # concatenate these arrays to get a dataset for background
    bg_data = np.hstack((side_low, side_high))

    # render histogram data
    mass_iqr = stats.iqr(region1)           # here we use region1 to maintain consistency and avoid different sized bins being used
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)
    bg_counts, bg_masses = np.histogram(bg_data, bins=num_bins, range=[np.min(bg_data), np.max(bg_data)])

    # get rid of empty regions 
    bg_masses = bg_masses[np.where(bg_counts != 0)]
    bg_counts = bg_counts[np.where(bg_counts != 0)]

    return bg_masses, bg_counts, bg_data

bg_masses = background()[0]
bg_all = background()[2]
bg_counts = background()[1]

# this is only the peak data for the histogram
# note edge 1 and edge 2 are just estimates. we need to use Niamh's scientific method here
peak_data = xmass[np.where((xmass > edge1) & (xmass < edge2))]

def plot_bg():
    plot_histogram(xmass_name, bg_all, xmass_units)



def fit_bg():
    '''
    this fits the background data to an exponential decay function
    returns the function variables a and b for a * np.exp(-b*t)
    '''
    popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(-b*t),  bg_masses,  bg_counts, maxfev=90000)

    # random dataset to display exponential decay function 
    x = np.linspace(np.min(bg_masses), np.max(bg_masses),1000)

    #pylab.plot(x, popt[0]*np.exp(-popt[1]*x),label='line of best fit')
    #pylab.plot(bg_masses, bg_counts, label='data')
    #pylab.legend()
    #pylab.show()
    return popt

#cool this works now we need to subtract this data from the original histogram

def remove_bg():
    #first we need to render histogram data for the whole of region1
    mass_iqr = stats.iqr(region1)
    bin_width = 2 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)
    all_counts, all_masses = np.histogram(region1, bins=num_bins, range=[np.min(region1), np.max(region1)])
    
    # create set of points to fit region1 
    x = np.linspace(np.min(all_masses), np.max(all_masses),len(all_masses)-1)
    popt = fit_bg()
    ydata = popt[0]*np.exp(-popt[1]*x)
    clear_data = all_counts - ydata

    '''
    We can plot this stuff to visualise our cleared signal

    pylab.plot(all_masses[0:-1], clear_data, label='cleared signal')
    pylab.plot(all_masses[0:-1], all_counts, label='all signals')
    pylab.plot(bg_masses, bg_counts, label='background data')
    pylab.plot(x, ydata, label='best fit function')
    pylab.title("Histogram of " + str(xmass_name) + " data" )
    pylab.xlabel(r'$M ( \mu ^-\mu ^+)$ $(GeV/c^2)$')
    pylab.ylabel(r'Candidates/ (25 Mev$/c^2$)')
    pylab.xlim(np.min(region1), np.max(region1))
    pylab.ylim(0)
    pylab.legend()
    pylab.show()
    '''

    return all_masses[0:-1], clear_data

def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def fit_gaussian():
    '''
    This fits the first peak to a normal distribution
    It doesn't work yet
    '''
    x, y = remove_bg()

    n = len(x)                          #the number of data
    mean = sum(x*y)/n                   
    sigma = np.sqrt(sum(y*(x-mean)**2)/n)

    popt2, pcov2 = curve_fit(gaus, x, y, p0=[max(y), 9.4551, sigma], maxfev=900000)
    a, x0, sigma = popt2

    print(popt2)
    pylab.plot(x, gaus(x, a, x0, sigma))
    pylab.plot(x,y)
    pylab.show()
fit_gaussian()




