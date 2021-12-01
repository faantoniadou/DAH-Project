# Data Analysis and Handling Project Report (F1):
# Comparison of Maximum likelihood fits as a means of calculating the masses of the Υ(1S), Υ(2S) and Υ(3S) meson states. 

## Overview 

This project aims to analyse the LHCb data sample collected in April 2015, at centre-of-mass energy √s = 13 Tev on the invariant mass of particle candidates. The dataset records measurements for 4,689,467 selected events. For each event, measurements of dimuon invariant masses, dimuon transverse momentum, dimuon rapidity, dimuon momentum, and the individual momenta of each muon of the pair. To do so, we use non-linear least squares and namely and the binned data to obtain best-fit parameters for our shallow exponential background curve. We namely use the scipy.optimise.curve fit minimisation engine provided by the SciPy open source library which employs the Levenberg-Marquardt algorithm. We first used a single Gaussian model, followed by an overlapping Gaussians model in order to improve accuracy of calculated masses.

Potential improvements in our method include:
1. Using Minuit, which has a number of advantages over scipy.curve_fit including the fact hat it is not susceptible to binning and hence it disregards random fluctuations. Hence, there is improved differentiation between background and signal events, the ability to account for errors with more precision, and to examine correlations between the discriminating variables employed in the analysis.
    
2. Using CrystalBall functions to fit our peaks instead of overlapped gaussians as this could potentially decrease the magnitude of residuals and improve their spread.
Adapt an outlier detection method to perform cuts in signal data, by considering data from kinematic properties of particles (e.g. higher/lower momenta particles are more likely to be contamination). This removes unwanted background from the mass spectrum. After performing these cuts, a different exponential fit can be used to model the remaining background.

3. Adapt an outlier detection method to perform cuts in signal data, by considering data from kinematic properties of particles (e.g. higher/lower momenta particles are more likely to be contamination). This removes unwanted background from the mass spectrum which are events accepted by trigger but are probably not upsilon candidates. After performing these cuts, a different exponential fit can be used to model the remaining background.

4. Change the region where exp background is estimated to be further away from signal regions. This will ensure that only background events are considered when performing an exponential fit to the background, even though less data points will be inputted in our minimizer.

## Usage
Our repository (https://github.com/faantoniadou/DAH-Project) includes files which perform fits for either of the two models, separately for each peak and for all peaks simultaneously. Generally, functions are repetitive amongst each file, so for **full commenting and documentation please turn to the simultaneous_bimodal_fit.py file**. The breakdown below explains the structure:
### Single Gaussian fits:
1. All Peaks: `simultaneous_gaussian_fit.py`
2. Peak 1: `peak_1_analysis.py`
3. Peak 2: `residuals 2.py`
4. Peak 3: `residuals 3.py`

### Overlapping Gaussian fits:
1. All Peaks: `simultaneous_bimodal_fit.py`
2. Peak 1: `bimodal_peak 1.py`
3. Peak 2: `bimodal residuals 2.py`
4. Peak 3: `bimodal residuals 3.py`

In each of these files, we use an exponential fit to remove the background and subsequently fit the remaining data using either a single Gaussian or overlapping Gaussians model. Most plotted graphs can be found in our Project Report.

Further analysis was done on relative production rates of Y(S1), Y(S2), Y(S3) by transverse momentum. This analysis allows for most theoretical and experimental uncertainties to cancel, facilitating us to test the underlying model with greater accuracy. The analysis can be found in `ratio_code.py`.

To find peak widths we used a peak finding method included in **peak_finding.py** as an approximation. These were later modified to yield best results in the analysis.
Histograms of all data properties can be plotted through `histogram_plots.py`

**All abovementioned files can be downloaded from our repository and ran separately on any scientific environment for Python.**

## Contributors 
This project exists thanks to Faidra Antoniadou and Niamh Clarke who contributed.

<a href="https://github.com/faantoniadou"><img src="https://avatars.githubusercontent.com/u/63123583?v=4" /></a>
<a href="https://github.com/niamhyclarke"><img src="https://avatars.githubusercontent.com/u/72151616?v=4" /></a>


