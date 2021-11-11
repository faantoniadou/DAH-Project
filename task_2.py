"""
function finding code
"""

import  numpy  as  np
import pylab
from scipy import stats
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


#  import  data
#xmass  =  np.loadtxt(sys.argv[1])


def plot_histogram(name, values, units)  :     
    #find binwidth, use Freeman-Diaconis rule
    mass_iqr = stats.iqr(values)
    bin_width = 5 * mass_iqr/((nevent)**(1/3))    
    num_bins = int(2/bin_width)
    pylab.hist(values,  bins=num_bins,  range=[np.min(values), np.max(values)])
    pylab.title("Histogram of " + name + " data" )
    pylab.ylabel("Counts in bin")
    pylab.xlabel(name + " " + units)
    print(bin_width)
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
'''

def background():
    # number of events
    nevents = len(region1)
    #print(N_X)

    # define sideband regions each be half as wide as the signal region 
    side_low = xmass[np.where((xmass < 9.29) & (xmass > 9.0))]
    side_high = xmass[np.where((xmass < 9.75) & (xmass > 9.61))]

    # concatenate these arrays to get a dataset for background
    bg_data = np.hstack((side_low, side_high))

    # render histogram data
    bg_counts, bg_masses = np.histogram(bg_data, bins=1000, range=[np.min(bg_data), np.max(bg_data)])

    # get rid of empty regions 
    bg_masses = bg_masses[np.where(bg_counts != 0)]
    bg_counts = bg_counts[np.where(bg_counts != 0)]

    return bg_masses, bg_counts, bg_data


# this is the function form that we want the data to be fitted to 
def exp_decay(data, a, b, c):
    return a * np.exp(-b * data) + c

bg_data = background()[0]
bg_all = background()[2]

def plot_bg():
    plot_histogram(xmass_name, bg_all, xmass_units)

plot_bg()
ydata = background()[1]
popt, pcov = curve_fit(lambda t, a, b: a * np.exp(b*t),  bg_data,  ydata, maxfev=5000)
print(popt)
