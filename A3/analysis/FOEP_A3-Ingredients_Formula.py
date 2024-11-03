import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import uncertainties as uct
from uncertainties import unumpy as unp

def adduct(arr, uarr):
    """Given array and corresponding uncertainty array, return unumpy array"""
    n = len(arr)
    result = np.array([])
    for i in range(n):
        result = np.append(result, uct.ufloat(arr[i], uarr[i]))
    return result

def uctarravg(arr):
    """Given unumpy array, return average (ufloat)"""
    n = len(arr)
    result = uct.ufloat(0, 0)
    for i in range(n):
        result += arr[i]    
    result /= n
    return result

def linreg(xarr, yarr):
    """Given unumpy arrays x and y, return ufloats slope and interception"""
    n = len(xarr)
    xavg = uctarravg(xarr)
    yavg = uctarravg(yarr)
    sxy = uct.ufloat(0, 0)
    sxx = uct.ufloat(0, 0)
    for i in range(n):
        sxy += (xarr[i] - xavg) * (yarr[i] - yavg)
        sxx += (xarr[i] - xavg)**2
    slope = sxy / sxx
    interception = yavg - slope * xavg
    return slope, interception

def epn(x):
    """Return the exponent of x"""
    return int(math.floor(math.log10(abs(x))))

def sigfig(x, d):
    """Return the d-th significant figure of x"""
    return round(x, -epn(x) + (d - 1))

def sigceil(u, d):
    """Ceiling u to 2nd significant figure"""
    tmp = 10 ** (-epn(u) + (d - 1))
    return math.ceil(u * tmp) / tmp

def errdat(x):
    """Print ufloat x to the precision of the 2nd significant figure of the error of x"""
    val, err= x.n, x.s
    uct.ufloat(sigfig(val, -epn(err) + 2), sigceil(err, 2))

# print(repr(errdat(uct.ufloat(2.35847357835, 0.000032483848))))

yo = 225.81
ndo = 336.48
baco = 197.3392
cuo = 79.5454

ymol = 2 / (yo + 4 * baco + 6 * cuo)
myo = ymol * yo
mybaco = ymol * 4 * baco
mycuo = ymol * 6 * cuo
print(myo, mybaco, mycuo)

ndmol = 2 / (ndo + 4 * baco + 6 * cuo)
mndo = ndmol * yo
mndbaco = ndmol * 4 * baco
mndcuo = ndmol * 6 * cuo
print(mndcuo, mndbaco, mndcuo)

df = pd.DataFrame({'Y2O3mol' : [ymol], 
                   'Y2O3mass' : [myo], 
                   'YBaCO3mass' : [mybaco], 
                   'YCuOmass' : [mycuo], 
                   'Nd2O3mol' : [ndmol], 
                   'Nd2O3mass' : [mndo], 
                   'NdBaCO3mass' : [mndbaco], 
                   'NdCuOmass' : [mndcuo]})
df.to_csv('../data/FOEP_A3-II-Ingredients_Formula.csv', index=False)
