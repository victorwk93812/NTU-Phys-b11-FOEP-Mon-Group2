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

## Superconducting and Critical Temperature

# Opening files
rawdata = ["../data/FOEP_A4-YBCO.txt", "../data/FOEP_A4-NBCO.txt"]
datafile = [open(rawdata[0], 'r'), open(rawdata[1], 'r')]

# Building raw data array, temperature array and resistance array
# 0: YBCO, 1: NBCO
dataarr = [0, 0]
temparr = [0, 0]
resarr = [0, 0]
for i in range(2):
    tmpdataarr = np.empty(shape = [0, 2])
    tmptemparr = np.array([])
    tmpresarr = np.array([])
    for line in datafile[i]:
        strings = line.split()
        temp, res = float(strings[0]), float(strings[1])
        point = np.array([[temp, res]])
        tmpdataarr = np.append(tmpdataarr, point, axis = 0)
        tmptemparr = np.append(tmptemparr, [temp])
        tmpresarr = np.append(tmpresarr, [res])
    dataarr[i] = tmpdataarr
    temparr[i] = tmptemparr
    resarr[i] = tmpresarr

datasize = [len(temparr[0]), len(temparr[1])]

# Original R-T graph
fig1, axs1 = plt.subplots(1, 2, figsize = (12, 4))
axs1[0].set_title("YBCO R-T")
axs1[1].set_title("NBCO R-T")
for i in range(2):
    axs1[i].set_xlabel(r'Temperature (K)')
    axs1[i].set_ylabel(r'Resistance (m$\Omega$)')
for i in range(2):
    axs1[i].plot(temparr[i], resarr[i])
fig1.savefig("../pics/R-T_Graph.png")

# Smoothing graphs
resarr[0] = savgol_filter(resarr[0], window_length=19, polyorder=1)
resarr[1][0:470] = savgol_filter(resarr[1][0:470], window_length=21, polyorder=2)
resarr[1][0:470] = medfilt(resarr[1][0:470], kernel_size=51)
resarr[1][300:880] = savgol_filter(resarr[1][300:880], window_length=95, polyorder=2)
temparr[1][0:470] = medfilt(temparr[1][0:470], kernel_size=51)
temparr[1][900:1500] = medfilt(temparr[1][900:1500], kernel_size=81)
temparr[1][1500:1800] = savgol_filter(temparr[1][1500:1800], window_length=51, polyorder=2)
temparr[1][1500:1800] = medfilt(temparr[1][1500:1800], kernel_size=11)
resarr[1][1500:1800] = savgol_filter(resarr[1][1500:1800], window_length=51, polyorder=2)

# Plotting smoothed graphs
fig2, axs2 = plt.subplots(1, 2, figsize = (12, 4))
axs2[0].set_title("YBCO R-T Smoothed")
axs2[1].set_title("NBCO R-T Smoothed")
for i in range(2):
    axs2[i].set_xlabel(r'Temperature (K)')
    axs2[i].set_ylabel(r'Resistance (m$\Omega$)')

for i in range(2):
    axs2[i].plot(temparr[i], resarr[i])
fig2.savefig("../pics/R-T_Graph-smoothed.png")

# Building YBCO dR/dT and time index arrays
space = 5
slopearr = [0]
timeind = [0]
tmpslopearr = np.array([])
tmptimeind = np.array([])
for j in range(space, datasize[0] - space - 1, (2 * space + 1)):
    slopeavg = 0
    cnt = 0
    for k in range(space + 1):
        tempdiff = temparr[0][j - k + space] - temparr[0][j - k]
        resdiff = resarr[0][j - k + space] - resarr[0][j - k]
        if tempdiff != 0:
            slope = resdiff / tempdiff
            cnt += 1
        slopeavg += slope
    slopeavg /= cnt
    tmpslopearr = np.append(tmpslopearr, [slopeavg])
    tmptimeind = np.append(tmptimeind, j)
slopearr = tmpslopearr
timeind = tmptimeind

# Plotting dR/dT-time graph
fig3, axs3 =plt.subplots(1, figsize = (6, 4))
axs3.set_title(r"YBCO dR/dT-Time")
axs3.set_xlabel(r"Time")
axs3.set_ylabel(r"dR/dT")
axs3.plot(timeind, slopearr)
fig3.savefig('../pics/YBCO-dRdT-Time_Graph.png')
plt.show()

# Observed two data points representing curie points while temperature 
# increasing and decreasing. 
# Find the two points and print
for i in range(len(timeind)):
    if temparr[0][int(timeind[i])] > 100:
        continue
    slopemax = 0
    for j in range(i - 5, i + 5):
        slopemax = max(slopemax, slopearr[j])
    if slopearr[i] == slopemax and slopearr[i] > 0.025:
        print("Critical point:", temparr[0][int(timeind[i])], "K, Time index:", int(timeind[i]))

# Closing files
datafile[0].close()
datafile[1].close()


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

def rnddec(x, d):
    """Round x to d decimal places"""
    # return round(x, -epn(x) + (d - 1))
    return round(x, (d - 1))

def sigfig(x, d):
    """Return the d-th significant figure of x"""
    return round(x, -epn(x) + (d - 1))

def sigceil(u, d):
    """Ceiling u to d-th significant figure"""
    tmp = 10 ** (-epn(u) + (d - 1))
    return math.ceil(u * tmp) / tmp

def errdat(x):
    """Print ufloat x to the precision of the 2nd significant figure of the error of x"""
    val, err= x.n, x.s
    return uct.ufloat(rnddec(val, -epn(err) + 2), sigceil(err, 2))

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
        print(sample, resname, ':', repr(errdat(resistance[i][j])), "Omega")

# Solved f using an electronic calculator additionally
f = np.array([0.7971806, 0.6855196])

# Resistivity
rho = np.array([uct.ufloat(0, 0), uct.ufloat(0, 0)])
for i in range(2):
    sample = "YBCO" if i == 0 else "NBCO"
    print(sample, "RCDBA/RBCDA =", resistance[i][0]/resistance[i][1])
    print(sample, "f =", f[i])
    rho[i] = (math.pi * samplewidth[i] * f[i] * (resistance[i][0] + resistance[i][1]))/(2 * math.log(2))
    print(sample, "rho =", rho[i], "(Omega mm)")
