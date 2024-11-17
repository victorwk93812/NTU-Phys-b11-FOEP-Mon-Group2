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
    """Return ufloat x to the precision of the 2nd significant figure of the error of x"""
    val, err= x.n, x.s
    return uct.ufloat(round_decimal_places(val, -exponent(err) + 2), ceiling_significant_figure(err, 2))

def ufloat_print_format(x):
    """Return formatted string form of ufloat x to the precision of the 2nd significant figure of the error of x"""
    x = ufloat_format(x)
    d = 2 # precision
    precision = abs(exponent(x.s)) + d - 1
    valstr, errstr = f'{x.n:.{precision}f}', f'{x.s:.{precision}f}'
    return f"{valstr}({errstr[-2:]})"

### Steam Method ###

ds = "../data/A6-2.csv"
df = pd.read_csv(ds)
# Micrometer error (mm)
micerr = df["IceSiherr(mm)"][0]
# Vernier error (cm)
vererr = df["Icederr(cm)"][0]
# Thermometer error (K)
thmerr = df["IceTerr(K)"][0]
# Mass error (g)
maserr = df["Icemerr(g)"][0]
# Sample width (mm)
samwid = df["IceSih(mm)"][0]
# Ice heat capacity (J/g)
cap = df["IceC(J/g)"][0]
# Trial numbering
trial = [2, 3]
time = np.array([df[f"Ice{tr}Deltat(s)"][0] for tr in trial])
t1 = np.array([uct.ufloat(df[f"Ice{tr}T1(C)"][0], thmerr) for tr in trial])
t2 = np.array([uct.ufloat(df[f"Ice{tr}T2(C)"][0], thmerr) for tr in trial])
m1 = np.array([uct.ufloat(df[f"Ice{tr}m1(g)"][0], maserr) for tr in trial])
m2 = np.array([uct.ufloat(df[f"Ice{tr}m2(g)"][0], maserr) for tr in trial])
d1 = np.array([uct.ufloat(df[f"Ice{tr}d1(cm)"][0], vererr) for tr in trial])
d2 = np.array([uct.ufloat(df[f"Ice{tr}d2(cm)"][0], vererr) for tr in trial])
# Sample width (m)
h = uct.ufloat(samwid, micerr) / 1000
# Temperature difference (K)
dt = t2 - t1
# Mass difference (g)
dm = m2 - m1
# Heat transferred (J)
heat = dm * cap
# Ice surface area (m^2)
A = np.pi * (((d1 + d2) / 2) ** 2) / 4 / 10000
K = h * heat / (A * dt * time)
print(f"Si Sample width is {ufloat_print_format(h * 1000)} mm.\n")
for i in range(2):
    print(f"{time[i]}s Experiment T difference is {ufloat_print_format(dt[i])} K.\n")
    print(f"{time[i]}s Experiment mass difference is {dm[i]} g.\n")
    print(f"{time[i]}s Experiment heat transferred is {ufloat_print_format(heat[i])} J.\n")
    print(f"{time[i]}s Experiment ice surface area is {ufloat_print_format(A[i] * 10000)} cm^2.\n")
    print(f"{time[i]}s Experiment sample thermal conductivity is {ufloat_print_format(K[i])} W/mK.\n")
