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

def ufloat_print_format(x):
    """Return formatted string form of ufloat x to the precision of the 2nd significant figure of the error of x"""
    x = ufloat_format(x)
    d = 2 # precision
    precision = abs(exponent(x.s)) + d - 1
    valstr, errstr = f'{x.n:.{precision}f}', f'{x.s:.{precision}f}'
    return f"{valstr}({errstr[-2:]})"

def ufloat_align_error_precision(x, d):
    """Given a ufloat x, return strings x.n and x.s with x.n formatted such that 
    the last digit of them are aligned to the d-th significant figure of x.n"""
    precision = abs(exponent(x.s)) + d - 1
    return f'{x.n:.{precision}f}', f'{x.s:.{precision}f}'


### Seeback Coefficient ###

ds = "../data/A6-1.csv"
df = pd.read_csv(ds)
# Materials
mat = ["Si", "Bi2Te3"]
# Temperature inc/dec
trd = ["inc", "dec"]
fulltrd = ["increasing", "decreasing"]
Verr = [df["Volterr(mV)"][0] for _ in range(5)]
Terr = [df["Terr(C)"][0] for _ in range(5)]
Ierr = [df["Ierr(mA)"][0] for _ in range(5)]
# Temperature array (C), T[i][j]: Material mat[i], j:increasing, decreasing
T = [[np.array(df[f"{mat[i]}DeltaT{trd[j]}(C)"]) for j in range(2)] for i in range(2)]
# Temperature array with uncertainties
Tuct = [[unp.uarray(np.array(df[f"{mat[i]}DeltaT{trd[j]}(C)"]), Terr) for j in range(2)] for i in range(2)]
# Voltage array (mV), T[i][j]: Material mat[i], j:increasing, decreasing
V = [[np.array(df[f"{mat[i]}DeltaV{trd[j]}(mV)"]) for j in range(2)] for i in range(2)]
# Voltage array with uncertainties
Vuct = [[unp.uarray(np.array(df[f"{mat[i]}DeltaV{trd[j]}(mV)"]), Verr) for j in range(2)] for i in range(2)]
S = [[- linear_regression(Tuct[i][j], Vuct[i][j])[0] for j in range(2)] for i in range(2)]
interception = [[linear_regression(Tuct[i][j], Vuct[i][j])[1] for j in range(2)] for i in range(2)]
fig, axs = plt.subplots(2, 2, figsize = (10, 10))
for i in range(2):
    for j in range(2):
        # print(f"Seeback coefficient of {mat[i]} in temperature {fulltrd[j]} is {ufloat_print_format(S[i][j])} mV/K.\n")
        print(f"Seeback coefficient of {mat[i]} in temperature {fulltrd[j]} is {S[i][j]} mV/K.\n")
        Tsample = np.linspace(T[i][j][0], T[i][j][-1], 100)
        fitcurve = (- S[i][j].n) * Tsample + interception[i][j].n
        axs[i][j].scatter(T[i][j], V[i][j], label = "Data points")
        axs[i][j].plot(Tsample, fitcurve, color = 'red', label = "Fitted Curve")
        axs[i][j].grid()
        axs[i][j].legend()
        axs[i][j].set_title(f"{mat[i]} Temperature {fulltrd[j]} V-T Graph")
fig.suptitle("Seeback Coefficients Plots")
plt.show()
