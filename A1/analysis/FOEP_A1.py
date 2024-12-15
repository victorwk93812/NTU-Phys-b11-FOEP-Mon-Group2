import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import uncertainties as uct
from uncertainties import unumpy as unp
import scipy
from scipy.optimize import curve_fit, fsolve
from scipy.signal import savgol_filter, medfilt
import sympy as sp
from sympy.solvers import solve
from sympy import Symbol

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

raw = ["../data/FOEP_A1-1.csv", "../data/FOEP_A1-2.csv"]
num = len(raw)
# print("Hello World!")
dfs = [pd.read_csv(path) for path in raw]
res = [uct.ufloat(df["R(omega)"][0], df["uR(omega)"][0]) * 1000 for df in dfs]
resBCDAcirc = [uct.ufloat(df["RBCDAcirc(momega)"][0], df["uRBCDAcirc(momega)"][0]) for df in dfs]
resCDABcirc = [uct.ufloat(df["RCDABcirc(momega)"][0], df["uRCDABcirc(momega)"][0]) for df in dfs]
l = [uct.ufloat(df["len(mm)"][0], df["ulen(mm)"][0]) for df in dfs]
d = [uct.ufloat(df["d(A)"][0], df["ud(A)"][0]) * 1e-7 for df in dfs]
# Standard copper sample width 0.1mm
dstd = 0.1
w = [uct.ufloat(df["w(mm)"][0], df["uw(mm)"][0]) for df in dfs]
a = [d[i] * w[i] for i in range(len(d))]
resBCDAstd = [uct.ufloat(df["RBCDAstd(momega)"][0], df["uRBCDAstd(momega)"][0]) for df in dfs]
resCDABstd = [uct.ufloat(df["RCDABstd(momega)"][0], df["uRCDABstd(momega)"][0]) for df in dfs]
resACBDp = [uct.ufloat(df["RACBD+(momega)"][0], df["uRACBD+(momega)"][0]) for df in dfs]
resACBDn = [uct.ufloat(df["RACBD-(momega)"][0], df["uRACBD-(momega)"][0]) for df in dfs]

# Insect mask restivity
restivins = [res[i] * a[i] / l[i] for i in range(num)]
# print(restivins)
def VDPsolveres(resCDAB, resBCDA, d):
    """Calculate restivity using Van der Pauw method"""
    # print(resCDAB, resBCDA, d)
    def equation(x):
        return np.exp(- np.pi * resCDAB * d / x) + np.exp(- np.pi * resBCDA * d / x) - 1
    sol = fsolve(equation, 0.001)
    return sol[0]

# Restivity of circle sample
restivcirc = [VDPsolveres(resCDABcirc[i].n, resBCDAcirc[i].n, d[i].n) for i in range(num)]
# print(restivcirc)

# Restivity of standard copper sample
restivstd = [VDPsolveres(resCDABstd[i].n, resBCDAstd[i].n, dstd) for i in range(num)]
# print(restivstd)

magf = 0.23
mu = [abs(resACBDp[i] - resACBDn[i]) * d[i] / (magf * restivins[i]) * 10000 for i in range(num)]
# print(mu)
e = 1.6 * 1e-19
n = [1 / (e * (restivins[i] * 1e-4) * mu[i]) for i in range(num)]
# print(n)

for i in range(num):
    print(f"The resistivity of insect-shape masked Cu film with thickness {d[i] * 1e7} angstrom is {restivins[i] * 1e-6} omega m.")
    print(f"The resistivity of disk shaped Cu film with thickness {d[i] * 1e7} angstrom is {restivcirc[i] * 1e-6} omega m.")
    print(f"The resistivity of standard Cu sample with thickness {dstd} mm is {restivstd[i] * 1e-6} omega m.")
    print(f"The Hall mobility of insect-shape masked Cu film with thickness {dstd} mm is {mu[i]} cm^2 V^{{-1}} s^{{-1}}.")
    print(f"The carrier concentration of insect-shape masked Cu film with thickness {d[i]} mm is {n[i]} cm^{{-3}}.")






# nom, std = ufloat_align_error_precision(resistance[0][0], 2)
# print(ufloat_print_format(uct.ufloat(2.35847357835, 0.000032483848)))
# two_arrays_to_latex_table(uctarray1, uctarray2, name1, name2)
