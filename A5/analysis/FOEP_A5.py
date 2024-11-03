import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import uncertainties as uct
from uncertainties import unumpy as unp
import scipy
from scipy.optimize import curve_fit
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

def sort_data(angle, intensity):
    """Sort w.r.t angle"""
    angle = np.copy(angle)
    intensity = np.copy(intensity)
    n = len(angle)
    combined = np.empty([n, 2])
    for i in range(n):
        combined[i] = np.array([angle[i], intensity[i]])
    new_combined = sorted(combined, key = lambda entry: entry[0])
    for i in range(n):
        angle[i] = new_combined[i][0]
        intensity[i] = new_combined[i][1]
    return np.copy(angle), np.copy(intensity)

def gaussian(x, amplitude, mean, stddev):
    """Gaussian distribution"""
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def get_lattice_constant(wavelength, maxangle, h, k, l):
    """Given wavelength (angstrom), maxangle (degree), h, k, l, return lattice constant"""
    return wavelength * np.sqrt(h ** 2 + k ** 2 + l ** 2) / (2 * unp.sin(maxangle / 180 * np.pi))

def get_plane_distance(lattice_constant, h, k, l):
    """Calculate plane distance from lattice constant and (h, k, l)"""
    return lattice_constant / np.sqrt(h ** 2 + k ** 2 + l ** 2) 

def hkl_possibilities(wavelength, lattice_constant, angle):
    """Calculate possibilities of (h, k, l) given wavelength, lattice constant and max angle.\
            Implemented particularly for Potassium Alum"""
    hkl = []
    value = (2 * lattice_constant * np.sin(angle / 180 * np.pi) / wavelength) ** 2
    for h in range(0, 8):
        for k in range(h, 8):
            for l in range(k, 8):
                if abs(h ** 2 + k ** 2 + l ** 2 - value) <= 1:
                    hkl.append([h, k, l]) 
    return hkl

def get_FWHM(stddev):
    return 2 * abs(stddev) * np.sqrt(2 * np.log(2))

def get_crystallite_size(maxangle, FWHM, wavelength, kappa):
    return kappa * wavelength / ((FWHM / 180 * np.pi) * unp.cos(maxangle / 180 * np.pi))

# X-ray Wavelength
xray_wavelength = 1.5406

# Crystallite constant
kappa = 0.9

# Read csv as dataframe
datasource = ["../data/FOEP_A5-1.csv", "../data/FOEP_A5-2.csv"]
df = [pd.read_csv(datasource[0]), pd.read_csv(datasource[1])]

# 8 plots
fig, axs = plt.subplots(4, 2, figsize = (20, 20))
plt.subplots_adjust(left = 0.03, bottom = 0.03, top = 0.94, right = 0.97, wspace = 0.1, hspace = 0.3)


### MgO(200) ###

# Sample = [2 * platform angle, intensity, detect angle, guassian initial guess, optimal parameters, optimal covariance, hkl, lattice constant, angle peak with uncertainty, plane distance, FWHM, crystallite size]
MgO200 = [0, 0, 0, 0, 0, 0, [2, 0, 0], 0, 0, 0, 0, 0]
digit1 = np.copy(np.array(df[0]["MgO200_Theta_plat-1"]))
digit2 = np.copy(np.array(df[0]["MgO200_Theta_plat-2"]))
digit3 = np.copy(np.array(df[0]["MgO200_Theta_plat-3"]))
intensity = np.copy(np.array(df[0]["MgO200_Int(muA)"]))

# Detector angle
MgO200[2] = np.copy(np.array(df[0]["MgO200_2Theta_detect"][0]))

# Intensity
MgO200[1] = np.copy(intensity)

# 2 theta
MgO200[0] = 2 * (np.copy(digit1) + np.copy(digit2) / 60 + np.copy(digit3) / 3600)

# Remove NaN
MgO200[0] = MgO200[0][~np.isnan(MgO200[0])]
MgO200[1] = MgO200[1][~np.isnan(MgO200[1])]

# Sort by 2 theta
MgO200[0], MgO200[1] = sort_data(MgO200[0], MgO200[1])

# initial_guess = [amplitude, mean, stddev]
MgO200[3] = [90, 43.6, 0.7]

# optimized parameters, optimized covariance
MgO200[4], MgO200[5] = curve_fit(gaussian, MgO200[0], MgO200[1], p0=MgO200[3])

# theta peak
MgO200[8] = ufloat_format(uct.ufloat(MgO200[4][1], 2 * abs(MgO200[4][2])) / 2)

# Lattice Constant
MgO200[7] = get_lattice_constant(xray_wavelength, MgO200[8], *MgO200[6])

# Plane Distance
MgO200[9] = get_plane_distance(MgO200[7], *MgO200[6])

# FWHM
MgO200[10] = get_FWHM(MgO200[4][2] / 2)

# Crystallite size
MgO200[11] = get_crystallite_size(MgO200[8], MgO200[10], xray_wavelength, kappa)

# Reporting
print("The lattice constant of MgO is", f'{MgO200[7]:.2f}', "angstrom.")
print("The peak value of MgO(200) occurs at theta =", MgO200[8], "degrees.")
print("The distance between planes of MgO(200) is", MgO200[9], "angstrom.")
print(f"The FWHM of MgO(200) is {MgO200[10]:.3f} degrees.")
print(f"The crystallite size of MgO(200) is {MgO200[11]:.2f} angstrom.\n")

# Plots
axs[0][0].scatter(MgO200[0], MgO200[1], label="$MgO(200)$ Data")
x_space = np.linspace(MgO200[0][0], MgO200[0][-1], 100)
axs[0][0].plot(x_space, gaussian(x_space, *MgO200[4]), color="red", label="$MgO(200)$ Fitted Gaussian")

# Labelling x, y
axs[0][0].set_xlabel(r"$2 \theta (^\circ)$")
axs[0][0].set_ylabel(r"Intensity ($\mu$A)")

# Equation Display
lx, rx, ly, ry = *axs[0][0].get_xlim(), *axs[0][0].get_ylim()
dx, dy = (rx - lx), (ry - ly)
axs[0][0].set_title(r"$MgO(200)$ Peak X-Ray Diffraction")
equation = f'$y = {MgO200[4][0]:.2f} e^{{-\\frac{{(x - {MgO200[4][1]:.2f})^{{2}} }}{{ 2 \\cdot {MgO200[4][2]:.2f}^{{2}}}}}}$'
axs[0][0].text(lx + 0.5 * dx, ry - 0.5 * dy, equation, verticalalignment = 'center', horizontalalignment = 'center', fontsize = 16, bbox = {'facecolor': 'white'})

# Legend
axs[0][0].legend()

### Si(400) ###

Si400 = [0, 0, 0, 0, 0, 0, [4, 0, 0], 0, 0, 0, 0, 0]
digit1 = np.copy(np.array(df[0]["Si400_Theta_plat-1"]))
digit2 = np.copy(np.array(df[0]["Si400_Theta_plat-2"]))
digit3 = np.copy(np.array(df[0]["Si400_Theta_plat-3"]))
intensity = np.copy(np.array(df[0]["Si400_Int(muA)"]))
Si400[2] = np.copy(np.array(df[0]["Si400_2Theta_detect"][0]))
Si400[1] = np.copy(intensity)
Si400[0] = 2 * (np.copy(digit1) + np.copy(digit2) / 60 + np.copy(digit3) / 3600)
Si400[0] = Si400[0][~np.isnan(Si400[0])]
Si400[1] = Si400[1][~np.isnan(Si400[1])]
Si400[0], Si400[1] = sort_data(Si400[0], Si400[1])
Si400[3] = [85, 69, 0.7]
Si400[4], Si400[5] = curve_fit(gaussian, Si400[0], Si400[1], p0=Si400[3])
Si400[8] = ufloat_format(uct.ufloat(Si400[4][1], 2 * abs(Si400[4][2])) / 2)
Si400[7] = get_lattice_constant(xray_wavelength, Si400[8], *Si400[6])
Si400[9] = get_plane_distance(Si400[7], *Si400[6])
Si400[10] = get_FWHM(Si400[4][2] / 2)
Si400[11] = get_crystallite_size(Si400[8], Si400[10], xray_wavelength, kappa)
print("The lattice constant of Si is", f'{Si400[7]:.2f}', "angstrom.")
print("The peak value of Si(400) occurs at theta =", Si400[8], "degrees.")
print("The distance between planes of Si(400) is", Si400[9], "angstrom.")
print(f"The FWHM of Si(400) is {Si400[10]:.3f} degrees.")
print(f"The crystallite size of Si(400) is {Si400[11]:.2f} angstrom.\n")
axs[0][1].scatter(Si400[0], Si400[1], label="$Si(400)$ Data")
x_space = np.linspace(Si400[0][0], Si400[0][-1], 100)
axs[0][1].plot(x_space, gaussian(x_space, *Si400[4]), color="red", label="$Si(400)$ Fitted Gaussian")
axs[0][1].set_xlabel(r"$2 \theta (^\circ)$")
axs[0][1].set_ylabel(r"Intensity ($\mu$A)")
axs[0][1].set_title(r"$Si(400)$ Peak X-Ray Diffraction")
lx, rx, ly, ry = *axs[0][1].get_xlim(), *axs[0][1].get_ylim()
dx, dy = (rx - lx), (ry - ly)
equation = f'$y = {Si400[4][0]:.2f} e^{{-\\frac{{(x - {Si400[4][1]:.2f})^{{2}} }}{{ 2 \\cdot {Si400[4][2]:.2f}^{{2}}}}}}$'
axs[0][1].text(lx + 0.5 * dx, ry - 0.5 * dy, equation, verticalalignment = 'center', horizontalalignment = 'center', fontsize = 16, bbox = {'facecolor': 'white'})
axs[0][1].legend()

### LaAlO3(100) ###

LaAlO3100 = [0, 0, 0, 0, 0, 0, [1, 0, 0], 0, 0, 0, 0, 0]
digit1 = np.copy(np.array(df[0]["LaAlO3100_Theta_plat-1"]))
digit2 = np.copy(np.array(df[0]["LaAlO3100_Theta_plat-2"]))
digit3 = np.copy(np.array(df[0]["LaAlO3100_Theta_plat-3"]))
intensity = np.copy(np.array(df[0]["LaAlO3100_Int(muA)"]))
LaAlO3100[2] = np.copy(np.array(df[0]["LaAlO3100_2Theta_detect"][0]))
LaAlO3100[1] = np.copy(intensity)
LaAlO3100[0] = 2 * (np.copy(digit1) + np.copy(digit2) / 60 + np.copy(digit3) / 3600)
LaAlO3100[0] = LaAlO3100[0][~np.isnan(LaAlO3100[0])]
LaAlO3100[1] = LaAlO3100[1][~np.isnan(LaAlO3100[1])]
LaAlO3100[0], LaAlO3100[1] = sort_data(LaAlO3100[0], LaAlO3100[1])
LaAlO3100[3] = [70, 23.94, 0.7]
LaAlO3100[4], LaAlO3100[5] = curve_fit(gaussian, LaAlO3100[0], LaAlO3100[1], p0=LaAlO3100[3])
LaAlO3100[8] = ufloat_format(uct.ufloat(LaAlO3100[4][1], 2 * abs(LaAlO3100[4][2])) / 2)
LaAlO3100[7] = get_lattice_constant(xray_wavelength, LaAlO3100[8], *LaAlO3100[6])
LaAlO3100[9] = get_plane_distance(LaAlO3100[7], *LaAlO3100[6])
LaAlO3100[10] = get_FWHM(LaAlO3100[4][2] / 2)
LaAlO3100[11] = get_crystallite_size(LaAlO3100[8], LaAlO3100[10], xray_wavelength, kappa)
print("The lattice constant of LaAlO3 is", f'{LaAlO3100[7]:.2f}', "angstrom.")
print("The peak value of LaAlO3(100) occurs at theta =", LaAlO3100[8], "degrees.")
print("The distance between planes of LaAlO3(100) is", LaAlO3100[9], "angstrom.")
print(f"The FWHM of LaAlO3(100) is {LaAlO3100[10]:.3f} degrees.")
print(f"The crystallite size of LaAlO3(100) is {LaAlO3100[11]:.2f} angstrom.\n")
axs[1][0].scatter(LaAlO3100[0], LaAlO3100[1], label="$LaAlO_3(100)$ Data")
x_space = np.linspace(LaAlO3100[0][0], LaAlO3100[0][-1], 100)
axs[1][0].plot(x_space, gaussian(x_space, *LaAlO3100[4]), color="red", label="$LaAlO_3(100)$ Fitted Gaussian")
axs[1][0].set_xlabel(r"$2 \theta (^\circ)$")
axs[1][0].set_ylabel(r"Intensity ($\mu$A)")
axs[1][0].set_title(r"$LaAlO_3(100)$ Peak X-Ray Diffraction")
lx, rx, ly, ry = *axs[1][0].get_xlim(), *axs[1][0].get_ylim()
dx, dy = (rx - lx), (ry - ly)
equation = f'$y = {LaAlO3100[4][0]:.2f} e^{{-\\frac{{(x - {LaAlO3100[4][1]:.2f})^{{2}} }}{{ 2 \\cdot {LaAlO3100[4][2]:.2f}^{{2}}}}}}$'
axs[1][0].text(lx + 0.5 * dx, ry - 0.5 * dy, equation, verticalalignment = 'center', horizontalalignment = 'center', fontsize = 16, bbox = {'facecolor': 'white'})

axs[1][0].legend()

### LaAlO3(200) ###

LaAlO3200 = [0, 0, 0, 0, 0, 0, [2, 0, 0], 0, 0, 0, 0, 0]
digit1 = np.copy(np.array(df[0]["LaAlO3200_Theta_plat-1"]))
digit2 = np.copy(np.array(df[0]["LaAlO3200_Theta_plat-2"]))
digit3 = np.copy(np.array(df[0]["LaAlO3200_Theta_plat-3"]))
intensity = np.copy(np.array(df[0]["LaAlO3200_Int(muA)"]))
LaAlO3200[2] = np.copy(np.array(df[0]["LaAlO3200_2Theta_detect"][0]))
LaAlO3200[1] = np.copy(intensity)
LaAlO3200[0] = 2 * (np.copy(digit1) + np.copy(digit2) / 60 + np.copy(digit3) / 3600)
LaAlO3200[0] = LaAlO3200[0][~np.isnan(LaAlO3200[0])]
LaAlO3200[1] = LaAlO3200[1][~np.isnan(LaAlO3200[1])]
LaAlO3200[0], LaAlO3200[1] = sort_data(LaAlO3200[0], LaAlO3200[1])
LaAlO3200[3] = [50, 48.4, 0.7]
LaAlO3200[4], LaAlO3200[5] = curve_fit(gaussian, LaAlO3200[0], LaAlO3200[1], p0=LaAlO3200[3])
LaAlO3200[8] = ufloat_format(uct.ufloat(LaAlO3200[4][1], 2 * abs(LaAlO3200[4][2])) / 2)
LaAlO3200[7] = get_lattice_constant(xray_wavelength, LaAlO3200[8], *LaAlO3200[6])
LaAlO3200[9] = get_plane_distance(LaAlO3200[7], *LaAlO3200[6])
LaAlO3200[10] = get_FWHM(LaAlO3200[4][2] / 2)
LaAlO3200[11] = get_crystallite_size(LaAlO3200[8], LaAlO3200[10], xray_wavelength, kappa)
print("The lattice constant of LaAlO3 is", f'{LaAlO3200[7]:.2f}', "angstrom.")
print("The peak value of LaAlO3(200) occurs at theta =", LaAlO3200[8], "degrees.")
print("The distance between planes of LaAlO3(200) is", LaAlO3200[9], "angstrom.")
print(f"The FWHM of LaAlO3(300) is {LaAlO3200[10]:.3f} degrees.")
print(f"The crystallite size of LaAlO3(200) is {LaAlO3200[11]:.2f} angstrom.\n")
axs[1][1].scatter(LaAlO3200[0], LaAlO3200[1], label="$LaAlO_3(200)$ Data")
x_space = np.linspace(LaAlO3200[0][0], LaAlO3200[0][-1], 100)
axs[1][1].plot(x_space, gaussian(x_space, *LaAlO3200[4]), color="red", label="$LaAlO_3(200)$ Fitted Gaussian")
axs[1][1].set_xlabel(r"$2 \theta (^\circ)$")
axs[1][1].set_ylabel(r"Intensity ($\mu$A)")
axs[1][1].set_title(r"$LaAlO_3(200)$ Peak X-Ray Diffraction")
lx, rx, ly, ry = *axs[1][1].get_xlim(), *axs[1][1].get_ylim()
dx, dy = (rx - lx), (ry - ly)
equation = f'$y = {LaAlO3200[4][0]:.2f} e^{{-\\frac{{(x - {LaAlO3200[4][1]:.2f})^{{2}} }}{{ 2 \\cdot {LaAlO3200[4][2]:.2f}^{{2}}}}}}$'
axs[1][1].text(lx + 0.5 * dx, ry - 0.5 * dy, equation, verticalalignment = 'center', horizontalalignment = 'center', fontsize = 16, bbox = {'facecolor': 'white'})
axs[1][1].legend()

### LaAlO3(300) ###

LaAlO3300 = [0, 0, 0, 0, 0, 0, [3, 0, 0], 0, 0, 0, 0, 0]
digit1 = np.copy(np.array(df[0]["LaAlO3300_Theta_plat-1"]))
digit2 = np.copy(np.array(df[0]["LaAlO3300_Theta_plat-2"]))
digit3 = np.copy(np.array(df[0]["LaAlO3300_Theta_plat-3"]))
intensity = np.copy(np.array(df[0]["LaAlO3300_Int(muA)"]))
LaAlO3300[2] = np.copy(np.array(df[0]["LaAlO3300_2Theta_detect"][0]))
LaAlO3300[1] = np.copy(intensity)
LaAlO3300[0] = 2 * (np.copy(digit1) + np.copy(digit2) / 60 + np.copy(digit3) / 3600)
LaAlO3300[0] = LaAlO3300[0][~np.isnan(LaAlO3300[0])]
LaAlO3300[1] = LaAlO3300[1][~np.isnan(LaAlO3300[1])]
LaAlO3300[0], LaAlO3300[1] = sort_data(LaAlO3300[0], LaAlO3300[1])
LaAlO3300[3] = [70, 75.5, 0.7]
LaAlO3300[4], LaAlO3300[5] = curve_fit(gaussian, LaAlO3300[0], LaAlO3300[1], p0=LaAlO3300[3])
LaAlO3300[8] = ufloat_format(uct.ufloat(LaAlO3300[4][1], 2 * abs(LaAlO3300[4][2])) / 2)
LaAlO3300[7] = get_lattice_constant(xray_wavelength, LaAlO3300[8], *LaAlO3300[6])
LaAlO3300[9] = get_plane_distance(LaAlO3300[7], *LaAlO3300[6])
LaAlO3300[10] = get_FWHM(LaAlO3300[4][2] / 2)
LaAlO3300[11] = get_crystallite_size(LaAlO3300[8], LaAlO3300[10], xray_wavelength, kappa)
print("The lattice constant of LaAlO3 is", f'{LaAlO3300[7]:.2f}', "angstrom.")
print("The peak value of LaAlO3(300) occurs at theta =", LaAlO3300[8], "degrees.")
print("The distance between planes of LaAlO3(300) is", LaAlO3300[9], "angstrom.")
print(f"The FWHM of LaAlO3(300) is {LaAlO3300[10]:.3f} degrees.")
print(f"The crystallite size of LaAlO3(300) is {LaAlO3300[11]:.2f} angstrom.\n")
axs[2][0].scatter(LaAlO3300[0], LaAlO3300[1], label="$LaAlO_3(300)$ Data")
x_space = np.linspace(LaAlO3300[0][0], LaAlO3300[0][-1], 100)
axs[2][0].plot(x_space, gaussian(x_space, *LaAlO3300[4]), color="red", label="$LaAlO_3(300)$ Fitted Gaussian")
axs[2][0].set_xlabel(r"$2 \theta (^\circ)$")
axs[2][0].set_ylabel(r"Intensity ($\mu$A)")
axs[2][0].set_title(r"$LaAlO_3(300)$ Peak X-Ray Diffraction")
lx, rx, ly, ry = *axs[2][0].get_xlim(), *axs[2][0].get_ylim()
dx, dy = (rx - lx), (ry - ly)
equation = f'$y = {LaAlO3300[4][0]:.2f} e^{{-\\frac{{(x - {LaAlO3300[4][1]:.2f})^{{2}} }}{{ 2 \\cdot {LaAlO3300[4][2]:.2f}^{{2}}}}}}$'
axs[2][0].text(lx + 0.5 * dx, ry - 0.5 * dy, equation, verticalalignment = 'center', horizontalalignment = 'center', fontsize = 16, bbox = {'facecolor': 'white'})
axs[2][0].legend()

### Potassium Alum ###

KAS = [0, 0, 0, 0, 0, 0, [3, 3, 3], 0, 0, 0, 0, 0]
digit1 = np.copy(np.array(df[1]["KAS_Theta_plat-1"]))
digit2 = np.copy(np.array(df[1]["KAS_Theta_plat-2"]))
digit3 = np.copy(np.array(df[1]["KAS_Theta_plat-3"]))
intensity = np.copy(np.array(df[1]["KAS_Int(muA)"]))
KAS[2] = np.copy(np.array(df[1]["KAS_2Theta_detect"][0]))
KAS[1] = np.copy(intensity)
KAS[0] = 2 * (np.copy(digit1) + np.copy(digit2) / 60 + np.copy(digit3) / 3600)
KAS[0] = KAS[0][~np.isnan(KAS[0])]
KAS[1] = KAS[1][~np.isnan(KAS[1])]
KAS[0], KAS[1] = sort_data(KAS[0], KAS[1])
KAS[3] = [40, 38.06, 0.7]
KAS[4], KAS[5] = curve_fit(gaussian, KAS[0], KAS[1], p0=KAS[3])
KAS_lattice_constant = 12.157
print("The possibilities of Miller indices of Potassium Alum are\n")
KAS_hkl = hkl_possibilities(xray_wavelength, KAS_lattice_constant, KAS[4][1] / 2)
for i in range(len(KAS_hkl)):
    print(KAS_hkl[i])
KAS[8] = ufloat_format(uct.ufloat(KAS[4][1], 2 * abs(KAS[4][2])) / 2)
KAS[7] = get_lattice_constant(xray_wavelength, KAS[8], *KAS[6])
KAS[9] = get_plane_distance(KAS[7], *KAS[6])
KAS[10] = get_FWHM(KAS[4][2] / 2)
KAS[11] = get_crystallite_size(KAS[8], KAS[10], xray_wavelength, kappa)
print("\nThe lattice constant of Potassium Alum is", f'{KAS[7]:.2f}', "angstrom.")
print("The peak value of Potassium Alum(333) occurs at theta =", KAS[8], "degrees.")
print("The distance between planes of Potassium Alum(333) is", KAS[9], "angstrom.")
print(f"The FWHM of Potassium Alum(333) is {KAS[10]:.3f} degrees.")
print(f"The crystallite size of Potassium Alum(333) is {KAS[11]:.2f} angstrom.\n")
axs[2][1].scatter(KAS[0], KAS[1], label="Potassium Alum Data")
x_space = np.linspace(KAS[0][0], KAS[0][-1], 100)
axs[2][1].plot(x_space, gaussian(x_space, *KAS[4]), color="red", label="Potassium Alum Fitted Gaussian")
axs[2][1].set_xlabel(r"$2 \theta (^\circ)$")
axs[2][1].set_ylabel(r"Intensity ($\mu$A)")
axs[2][1].set_title(r"Potassium Alum(333) Peak X-Ray Diffraction")
lx, rx, ly, ry = *axs[2][1].get_xlim(), *axs[2][1].get_ylim()
dx, dy = (rx - lx), (ry - ly)
equation = f'$y = {KAS[4][0]:.2f} e^{{-\\frac{{(x - {KAS[4][1]:.2f})^{{2}} }}{{ 2 \\cdot {KAS[4][2]:.2f}^{{2}}}}}}$'
axs[2][1].text(lx + 0.5 * dx, ry - 0.5 * dy, equation, verticalalignment = 'center', horizontalalignment = 'center', fontsize = 16, bbox = {'facecolor': 'white'})
axs[2][1].legend()

### Si(440) ###

Si440 = [0, 0, 0, 0, 0, 0, [4, 4, 0], 0, 0, 0, 0, 0]
digit1 = np.copy(np.array(df[1]["Si440_Theta_plat-1"]))
digit2 = np.copy(np.array(df[1]["Si440_Theta_plat-2"]))
digit3 = np.copy(np.array(df[1]["Si440_Theta_plat-3"]))
intensity = np.copy(np.array(df[1]["Si440_Int(muA)"]))
Si440[2] = np.copy(np.array(df[1]["Si440_2Theta_detect"][0]))
Si440[1] = np.copy(intensity)
Si440[0] = 2 * (np.copy(digit1) + np.copy(digit2) / 60 + np.copy(digit3) / 3600)
Si440[0] = Si440[0][~np.isnan(Si440[0])]
Si440[1] = Si440[1][~np.isnan(Si440[1])]
Si440[0], Si440[1] = sort_data(Si440[0], Si440[1])
Si440[3] = [60, 17.2, 0.7]
Si440[4], Si440[5] = curve_fit(gaussian, Si440[0], Si440[1], p0=Si440[3])

# Should +45 degrees for Si(440) for computing the lattice constant
Si440[8] = ufloat_format(uct.ufloat(Si440[4][1], 2 * abs(Si440[4][2])) / 2 + 45)

Si440[7] = get_lattice_constant(xray_wavelength, Si440[8], *Si440[6])
Si440[9] = get_plane_distance(Si440[7], *Si440[6])
Si440[10] = get_FWHM(Si440[4][2] / 2)
Si440[11] = get_crystallite_size(Si440[8], Si440[10], xray_wavelength, kappa)
print("The lattice constant of Si is", f'{Si440[7]:.2f}', "angstrom.")

# Original data, deduct the 45 degrees back
print("The peak value of Si(440) occurs at theta =", Si440[8] - 45, "degrees.")

print("The distance between planes of Si(440) is", Si440[9], "angstrom.")
print(f"The FWHM of Si(440) is {Si440[10]:.3f} degrees.")
print(f"The crystallite size of Si(440) is {Si440[11]:.2f} angstrom.\n")
axs[3][0].scatter(Si440[0], Si440[1], label="$Si(440)$ Data")
x_space = np.linspace(Si440[0][0], Si440[0][-1], 100)
axs[3][0].plot(x_space, gaussian(x_space, *Si440[4]), color="red", label="$Si(440)$ Fitted Gaussian")
axs[3][0].set_xlabel(r"$2 \theta (^\circ)$")
axs[3][0].set_ylabel(r"Intensity ($\mu$A)")
axs[3][0].set_title(r"$Si(440)$ Peak X-Ray Diffraction")
lx, rx, ly, ry = *axs[3][0].get_xlim(), *axs[3][0].get_ylim()
dx, dy = (rx - lx), (ry - ly)
equation = f'$y = {Si440[4][0]:.2f} e^{{-\\frac{{(x - {Si440[4][1]:.2f})^{{2}} }}{{ 2 \\cdot {Si440[4][2]:.2f}^{{2}}}}}}$'
axs[3][0].text(lx + 0.5 * dx, ry - 0.5 * dy, equation, verticalalignment = 'center', horizontalalignment = 'center', fontsize = 16, bbox = {'facecolor': 'white'})
axs[3][0].legend()

### Si(404) ###

Si404 = [0, 0, 0, 0, 0, 0, [4, 0, 4], 0, 0, 0, 0, 0]
digit1 = np.copy(np.array(df[1]["Si404_Theta_plat-1"]))
digit2 = np.copy(np.array(df[1]["Si404_Theta_plat-2"]))
digit3 = np.copy(np.array(df[1]["Si404_Theta_plat-3"]))
intensity = np.copy(np.array(df[1]["Si404_Int(muA)"]))
Si404[2] = np.copy(np.array(df[1]["Si404_2Theta_detect"][0]))
Si404[1] = np.copy(intensity)
Si404[0] = 2 * (np.copy(digit1) + np.copy(digit2) / 60 + np.copy(digit3) / 3600)
Si404[0] = Si404[0][~np.isnan(Si404[0])]
Si404[1] = Si404[1][~np.isnan(Si404[1])]
Si404[0], Si404[1] = sort_data(Si404[0], Si404[1])
Si404[3] = [70, 16.68, 0.7]
Si404[4], Si404[5] = curve_fit(gaussian, Si404[0], Si404[1], p0=Si404[3])

# Should +45 degrees for Si(404) for computing the lattice constant
Si404[8] = ufloat_format(uct.ufloat(Si404[4][1], 2 * abs(Si404[4][2])) / 2 + 45)

Si404[7] = get_lattice_constant(xray_wavelength, Si404[8], *Si404[6])
Si404[9] = get_plane_distance(Si404[7], *Si404[6])
Si404[10] = get_FWHM(Si404[4][2] / 2)
Si404[11] = get_crystallite_size(Si404[8], Si404[10], xray_wavelength, kappa)
print("The lattice constant of Si is", f'{Si404[7]:.2f}', "angstrom.")

# Original data, deduct the 45 degrees back
print("The peak value of Si(404) occurs at theta =", Si404[8] - 45, "degrees.")

print("The distance between planes of Si(404) is", Si404[9], "angstrom.")
print(f"The FWHM of Si(404) is {Si404[10]:.3f} degrees.")
print(f"The crystallite size of Si(404) is {Si404[11]:.2f} angstrom.\n")
axs[3][1].scatter(Si404[0], Si404[1], label="$Si(404)$ Data")
x_space = np.linspace(Si404[0][0], Si404[0][-1], 100)
axs[3][1].plot(x_space, gaussian(x_space, *Si404[4]), color="red", label="$Si(404)$ Fitted Gaussian")
axs[3][1].set_xlabel(r"$2 \theta (^\circ)$")
axs[3][1].set_ylabel(r"Intensity ($\mu$A)")
axs[3][1].set_title(r"$Si(404)$ Peak X-Ray Diffraction")
lx, rx, ly, ry = *axs[3][1].get_xlim(), *axs[3][1].get_ylim()
dx, dy = (rx - lx), (ry - ly)
equation = f'$y = {Si404[4][0]:.2f} e^{{-\\frac{{(x - {Si404[4][1]:.2f})^{{2}} }}{{ 2 \\cdot {Si404[4][2]:.2f}^{{2}}}}}}$'
axs[3][1].text(lx + 0.5 * dx, ry - 0.5 * dy, equation, verticalalignment = 'center', horizontalalignment = 'center', fontsize = 16, bbox = {'facecolor': 'white'})
axs[3][1].legend()

# Figure title, saving and demo
fig.suptitle("X-ray Diffractions", fontsize = 30, y = 0.98)
fig.savefig("../pics/X-ray_Diffractions.png")
plt.show()
