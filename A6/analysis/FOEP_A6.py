import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import uncertainties as uct
from uncertainties import unumpy as unp
import scipy
from scipy.signal import savgol_filter, medfilt

def merge_array_uncertainties(arr, uarr):
    """Given array and corresponding uncertainty array, return unumpy array"""
    n = len(arr)
    result = np.array([])
    for i in range(n):
        result = np.append(result, uct.ufloat(arr[i], uarr[i]))
    return result

def array_average(arr):
    """Given number(unumpy) array, return average number(ufloat)"""
    n = len(arr)
    result = 0
    for i in range(n):
        result += arr[i]    
    result /= n
    return result

def linear_regression(xarr, yarr):
    """Given number(unumpy) arrays x and y, return numbers(ufloats) slope and interception"""
    n = len(xarr)
    xavg = array_average(xarr)
    yavg = array_average(yarr)
    sxy = uct.ufloat(0, 0)
    sxx = uct.ufloat(0, 0)
    for i in range(n):
        sxy += (xarr[i] - xavg) * (yarr[i] - yavg)
        sxx += (xarr[i] - xavg)**2
    slope = sxy / sxx
    interception = yavg - slope * xavg
    return slope, interception

def exponent(x):
    """Return the exponent of x"""
    return int(math.floor(math.log10(abs(x))))

def round_decimal_places(x, d):
    """Round x to d decimal places"""
    return round(x, (d - 1))

def significant_figure(x, d):
    """Return d-th significant figure of x"""
    return round(x, -exponent(x) + (d - 1))

def ceiling_significant_figure(u, d):
    """Ceiling u to d-th significant figure"""
    tmp = 10 ** (-exponent(u) + (d - 1))
    return math.ceil(u * tmp) / tmp

def ufloat_format(x):
    """Print ufloat x to the precision of the 2nd significant figure of the error of x"""
    val, err= x.n, x.s
    return uct.ufloat(round_decimal_places(val, -exponent(err) + 2), ceiling_significant_figure(err, 2))

def ufloat_align_error_precision(x, d):
    """Given a ufloat x, return strings x.n and x.s with x.n formatted such that 
    the last digit of them are aligned to the d-th significant figure of x.n"""
    precision = abs(exponent(x.s)) + d - 1
    return f'{x.n:.{precision}f}', f'{x.s:.{precision}f}'

def array_to_latex_table(x, xname):
    """Print an (ufloat) array x with its name into a latex table"""
    n = len(x)
    print(r'\begin{tabular}{M}')
    print("\t", xname, r'\\')
    print("\t", "\\hline")
    for i in range(n):
        if type(x[i]) is uct.core.Variables:
            dat = ufloat_format(x[i])
            nom, std = ufloat_align_error_precision(dat, 2)
            print("\t", nom, r"\pm", std, r"\\")
        elif type(x[i]) is float or type(x[i]) is np.float64:
            print("\t", f'{x[i]:.5f}', r"\\")
        else:
            print("\t", x[i], r"\\")
    print(r'\end{tabular}')

def two_arrays_to_latex_table(x, y, xname, yname):
    """Print (ufloat) arrays x and y with its name into a latex table"""
    if len(x) != len(y):
        return
    n = len(x)
    print(r'\begin{tabular}{MM}')
    print("\t", xname, '&', yname, r'\\')
    print("\t", "\\hline")
    for i in range(n):
        if type(x[i]) is uct.core.Variable:
            datx = ufloat_format(x[i])
            nomx, stdx = ufloat_align_error_precision(datx, 2)
            print("\t", nomx, r"\pm", stdx, r'&', end = ' ')
        elif type(x[i]) is float or type(x[i]) is np.float64:
            print("\t", f'{x[i]:.5f}', r'&', end = ' ')
        else:
            print("\t", x[i], r'&', end = ' ')
        if type(y[i]) is uct.core.Variable:
            daty = ufloat_format(y[i])
            nomy, stdy = ufloat_align_error_precision(daty, 2)
            print(nomy, r"\pm", stdy, r"\\")
        elif type(y[i]) is float or type(y[i]) is np.float64:
            print(f'{y[i]:.5f}', r"\\")
        else:
            print(type(y[i]), end = ' ')
            print(y[i], r"\\")
    print(r'\end{tabular}')

### Newton's Method ###

ntnumberlist = [2, 3, 6, 7]
ntdatasource = [f"../data/Newton_T{x}.csv" for x in ntnumberlist]
ntdf = [pd.read_csv(datsrc) for datsrc in ntdatasource]
nttime = [np.array(df["Time(s)"]) for df in ntdf]
nttemp = [np.array(df[f"Temperature{x}(C)"]) for x, df in zip(ntnumberlist, ntdf)]
# print(ntdf[3])
# print(nttime[1])

# Newton Plots
ntmergefig, ntmergeaxs= plt.subplots()
for i in range(4):
    ntmergeaxs.plot(nttime[i], nttemp[i], label = f"Newton T{ntnumberlist[i]}")
ntmergeaxs.legend()
plt.show()
# ntsepfig, ntsepaxs = plt.subplots()

### Angstrom's Method ###
angsmatname = ["Material 1", "Material 2"]
angsnumberlist = [1, 2, 7, 8]
angsdatasource = [f"../data/Angstrom_T{x}.csv" for x in angsnumberlist]
angsdf = [pd.read_csv(datsrc) for datsrc in angsdatasource]
angstime = [np.array(df["Time(s)"]) for df in angsdf]
angstemp = [np.array(df[f"Temperature{x}(C)"]) for x, df in zip(angsnumberlist, angsdf)]
# print(angsdf[3])
# print(angstime[1])

# Angstrom Plots
angsfigs, angsaxss = [0, 0], [0, 0]
for i in range(2):
    angsfigs[i], angsaxss[i] = plt.subplots()
for i in range(4):
    angsaxss[i // 2].plot(angstime[i], angstemp[i], label = f"Angstrom T{angsnumberlist[i]}")
for i in range(2):
    angsaxss[i].legend()
plt.show()


























# nom, std = ufloat_align_error_precision(resistance[0][0], 2)
# print(repr(ufloat_format(uct.ufloat(2.35847357835, 0.000032483848))))
# two_arrays_to_latex_table(uctarray1, uctarray2, name1, name2)
