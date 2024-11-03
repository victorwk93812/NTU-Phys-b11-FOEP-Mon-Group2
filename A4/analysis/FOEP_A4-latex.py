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

## Van der Paaw

# Utilities
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
    # sxy = uct.ufloat(0, 0)
    sxy = 0
    sxx = 0
    for i in range(n):
        sxy += (xarr[i] - xavg) * (yarr[i] - yavg)
        sxx += (xarr[i] - xavg)**2
    slope = sxy / sxx
    interception = yavg - slope * xavg
    return slope, interception

def exponent(x):
    """Return the exponent of x"""
    return int(math.floor(math.log10(abs(x))))

def rnddec(x, d):
    """Round x to d decimal places"""
    # return round(x, -exponent(x) + (d - 1))
    return round(x, (d - 1))

def sigfig(x, d):
    """Return the d-th significant figure of x"""
    return round(x, -exponent(x) + (d - 1))

def ceiling_significant_figure(u, d):
    """Ceiling u to d-th significant figure"""
    tmp = 10 ** (-exponent(u) + (d - 1))
    return math.ceil(u * tmp) / tmp

def format_ufloat(x):
    """Print ufloat x to the precision of the 2nd significant figure of the error of x"""
    val, err= x.n, x.s
    return uct.ufloat(rnddec(val, -exponent(err) + 2), ceiling_significant_figure(err, 2))

def align_error_precision(x, d):
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
            dat = format_ufloat(x[i])
            nom, std = align_error_precision(dat, 2)
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
            datx = format_ufloat(x[i])
            nomx, stdx = align_error_precision(datx, 2)
            print("\t", nomx, r"\pm", stdx, r'&', end = ' ')
        elif type(x[i]) is float or type(x[i]) is np.float64:
            print("\t", f'{x[i]:.5f}', r'&', end = ' ')
        else:
            print("\t", x[i], r'&', end = ' ')
        if type(y[i]) is uct.core.Variable:
            daty = format_ufloat(y[i])
            nomy, stdy = align_error_precision(daty, 2)
            print(nomy, r"\pm", stdy, r"\\")
        elif type(y[i]) is float or type(y[i]) is np.float64:
            print(f'{y[i]:.5f}', r"\\")
        else:
            print(type(y[i]), end = ' ')
            print(y[i], r"\\")
    print(r'\end{tabular}')

# Input data
data = []
rawdata = ["../data/FOEP_A4-1.csv", "../data/FOEP_A4-2.csv"]
for i in range(2): 
    data.append(pd.read_csv(rawdata[i]))

# Read current, voltage, their uncertainties and voltage bias
# First index: 
# 0: YBCO, 1: NBCO
# Second index: 
# 0: CDBA 1: BCDA
emptylist = [[0, 0], [0, 0]]
current = copy.deepcopy(emptylist)
ucurrent = copy.deepcopy(emptylist)
Vbias = copy.deepcopy(emptylist)
voltage = copy.deepcopy(emptylist)
uvoltage = copy.deepcopy(emptylist)
uctcurrent = copy.deepcopy(emptylist)
uctvoltage = copy.deepcopy(emptylist)
samplewidth = np.array([0.0, 0.0])
arrsize = 6

for i in range(2):
    current[i][0] = np.copy(np.array(data[i]["ICD(mA)"]))
    current[i][1] = np.copy(np.array(data[i]["IBC(mA)"]))
    ucurrent[i][0] = np.copy(np.array(data[i]["uICD(mA)"]))
    ucurrent[i][1] = np.copy(np.array(data[i]["uIBC(mA)"]))
    Vbias[i][0] = np.copy(np.array(data[i]["VbiasCDBA(mV)"]))
    Vbias[i][1] = np.copy(np.array(data[i]["VbiasBCDA(mV)"]))
    voltage[i][0] = np.copy(np.array(data[i]["VBA(mV)"]))
    voltage[i][1] = np.copy(np.array(data[i]["VDA(mV)"]))
    uvoltage[i][0] = np.copy(np.array(data[i]["uVBA(mV)"]))
    uvoltage[i][1] = np.copy(np.array(data[i]["uVDA(mV)"]))
    samplewidth[i] = data[i]["samplewidth(mm)"][0]

# Current and voltage with uncertainties
for i in range(2):
    for j in range(2):
        for k in range(arrsize):
            voltage[i][j][k] = voltage[i][j][k] - Vbias[i][j][0]
        uctcurrent[i][j] = np.copy(adduct(current[i][j], ucurrent[i][j]))
        uctvoltage[i][j] = np.copy(adduct(voltage[i][j], uvoltage[i][j]))

# Resistance
resistance = copy.deepcopy(emptylist)
for i in range(2):
    for j in range(2):
        sample = "YBCO" if i == 0 else "NBCO"
        resname = "RCDBA" if j == 0 else "RBCDA"
        resistance[i][j], tmp = linreg(uctcurrent[i][j], uctvoltage[i][j])
        print(sample, resname, ':', repr(format_ufloat(resistance[i][j])), "Omega")

# Solved f using an electronic calculator additionally
f = np.array([0.7971806, 0.6855196])
rho = np.array([uct.ufloat(0, 0), uct.ufloat(0, 0)])

# Resistivity
for i in range(2):
    sample = "YBCO" if i == 0 else "NBCO"
    print(sample, "RCDBA/RBCDA =", resistance[i][0]/resistance[i][1])
    print(sample, "f =", f[i])
    rho[i] = (math.pi * samplewidth[i] * f[i] * (resistance[i][0] + resistance[i][1]))/(2 * math.log(2))
    print(sample, "rho =", rho[i], "(Omega cm)")

for i in range(2):
    for j in range(2):
        two_arrays_to_latex_table(uctcurrent[i][j], uctvoltage[i][j], "I", "V")
for i in range(2):
    for j in range(2):
        two_arrays_to_latex_table(current[i][j], voltage[i][j], "I", "V")
