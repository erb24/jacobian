# utility functions required for running the Jacobian code. 

import numpy as np
from scipy import optimize
def list_enumerate(sequence, start=0):
    n = start
    for elem in sequence:
        yield n, *elem
        n += 1

def make_fes2d(x, y, bins = 50):
    hist, xbins, ybins = np.histogram2d(x, y, bins = bins)
    dx = xbins[1] - xbins[0]
    dy = ybins[1] - ybins[0]
    xbins = (xbins[1:] + xbins[:-1]) / 2
    ybins = (ybins[1:] + ybins[:-1]) / 2
    hist = hist / (hist.sum()*dx*dy)
    fes = -np.log(hist.T)
    return fes

def fuzzy_histogram(data, f, bin_centers, binwidth = 1):
    hist = np.zeros(len(bin_centers))
    counter = np.zeros_like(hist)
    for i, bin_center in enumerate(bin_centers):
        #print(i, bin_center)
        for k, x in enumerate(data):
            hist[i] += f[k] * np.exp(-(x - bin_center)**2/ (2 * binwidth*binwidth))
            counter[i] += np.exp(-(x - bin_center)**2/ (2 * binwidth*binwidth))
    return hist, counter

def fit_sigmoid(x, y):
    popt, pcov = optimize.curve_fit(sigmoid, x, y)
    a, b, c, d, e, f = popt; print(*popt)
    return sigmoid(x, a, b, c, d, e, f), popt

def sigmoid(x, a, b, c, d, e, f): 
    return a + (b - a) / ((c + d * np.exp(-e * x))**f)

def dsigmoid(x, a, b, c, d, e, f):
    return (b - a) * (f * e * d * np.exp(-e * x) * (1 + d * np.exp(-e * x))**(-f - 1))

def entropic_double_well_potential(x, y, sigma_x = 0.1, sigma_y = 0.1):
    return x**6 + y**6 + np.exp(-(y /sigma_y)**2) * (1 - np.exp(-(x / sigma_x)**2))

def a(x, delta = 0.05, x0 = 0.0):
    return 0.2*(1 + 5*np.exp(-(x - x0)**2 / delta))**2

def temperature_switch_potential(x, y, hx = 0.5, hy = 1.0, delta = 0.05, x0 = 0.0):
    return hx * (x**2 - 1)**2 + (hy + a(x, delta = delta, x0 = x0)) * (y**2 - 1)**2