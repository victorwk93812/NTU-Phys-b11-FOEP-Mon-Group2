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
# plt.show()

# Four points measurement
SiIpos = uct.ufloat(df["SiI+(mA)"][0], Ierr[0])
SiVpos = uct.ufloat(df["SiV+(mV)"][0], Verr[0])
SiIneg = uct.ufloat(df["SiI-(mA)"][0], Ierr[0])
SiVneg = uct.ufloat(df["SiV-(mV)"][0], Verr[0])
Bi2Te3Ipos = uct.ufloat(df["Bi2Te3I+(mA)"][0], Ierr[0])
Bi2Te3Vpos = uct.ufloat(df["Bi2Te3V+(mV)"][0], Verr[0])
Bi2Te3Ineg = uct.ufloat(df["Bi2Te3I-(mA)"][0], Ierr[0])
Bi2Te3Vneg = uct.ufloat(df["Bi2Te3V-(mV)"][0], Verr[0])

# Resistance
SiR = (SiVpos - SiVneg) / (SiIpos - SiIneg)
print(f"Resistance of Si is {ufloat_print_format(SiR)} (omega).\n")
Bi2Te3R = (Bi2Te3Vpos - Bi2Te3Vneg) / (Bi2Te3Ipos - Bi2Te3Ineg)
print(f"Resistance of Bi2Te3 is {ufloat_print_format(Bi2Te3R)} (omega).\n")
# h, w, l in mm
Sih, Siw, Sil = uct.ufloat(0.513, 0.0005), uct.ufloat(15, 2), uct.ufloat(30, 1)
Bi2Te3h, Bi2Te3w, Bi2Te3l = uct.ufloat(5, 0.02), uct.ufloat(10, 0.02), uct.ufloat(30, 1)
# Resistivity (omega m)
Sirho = SiR * Sih * Siw / Sil * 0.001
# Conductivity (omega^{-1} m^{-1})
Sisigma = 1 / Sirho
print(f"Resistivity of Si is {ufloat_print_format(Sirho)} (omega m).\n")
print(f"Conductivity of Si is {repr(Sisigma)} (omega^{{-1}} m^{{-1}}).\n")
Bi2Te3rho = Bi2Te3R * Bi2Te3h * Bi2Te3w / Bi2Te3l * 0.001
Bi2Te3sigma = 1 / Bi2Te3rho
print(f"Resistivity of Bi2Te3 is {ufloat_print_format(Bi2Te3rho)} (omega m).\n")
print(f"Conductivity of Bi2Te3 is {repr(Bi2Te3sigma)} (omega^{{-1}} m^{{-1}}).\n")

# Thermal conductivity from Angstrom
SiK = (uct.ufloat(41.1, 2.7) + uct.ufloat(17.6, 1.2)) / 2
Bi2Te3K = (uct.ufloat(19.1, 0.7) + uct.ufloat(3.33, 0.22)) / 2

# Average Seebeck coefficients (V/K)
SiS = (S[0][0] + S[0][1]) / 2 / 1000
Bi2Te3S = (S[1][0] + S[1][1]) / 2 / 1000

# Temperature (K)
T = 293

SiZT = (SiS**2) * Sisigma * T / SiK
Bi2Te3ZT = (Bi2Te3S**2) * Bi2Te3sigma * T / Bi2Te3K
print(f"Si figure of merit ZT: {ufloat_print_format(SiZT)}.")
print(f"Bi2Te3 figure of merit ZT: {ufloat_print_format(Bi2Te3ZT)}.")
