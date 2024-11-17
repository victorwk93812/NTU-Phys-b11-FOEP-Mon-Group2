import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import uncertainties as uct
from uncertainties import unumpy as unp
import scipy
from scipy.signal import savgol_filter, medfilt, get_window
from scipy.fft import fft, ifft, fftfreq

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

### Angstrom's Method ###
# Material names
Angsmatname = ["Si", "Bi2Te3"]
# Endpoint numbers
Angsendnum = [1, 2, 7, 8]
# Time, trial variables
Sitime = [30]
Sitrial = [1, 2]
Bi2Te3time = [40, 50, 60, 70, 80, 90]
# Heat source endpoints: T2, T7, response endpoints: T1, T8
endpts = [[2, 7], [1, 8]]
# Data source files: 0:T1, 1:T2, 2:T7, 3:T8
# Angs data source i from 0 to 7: 30-1, 30-2, 40, 50, 60, 70, 80, 90
# Angs data source j from 0 to 1: Tl, Tr: (l, r) = (1, 2) or (7, 8)
Sids = [[f"../data/Angstrom_T{num}_{Sitime[0]}s-{trial}.csv" for num in Angsendnum[:2]] for trial in Sitrial] 
Bi2Te3ds = [[f"../data/Angstrom_T{num}_{time}s.csv" for num in Angsendnum[2:]] for time in Bi2Te3time] 
Angsds = Sids + Bi2Te3ds
Angsdf = [[pd.read_csv(datsrc) for datsrc in endpointdat] for endpointdat in Angsds]

# Sidata i from 0 to 1: 30-1, 30-2
# Sidata j from 0 to 2: time, heat source T2, respond T1
Sidata = [[np.array(trialdata[1]["Time(s)"]), np.array(trialdata[1]["Temperature2(C)"]), np.array(trialdata[0]["Temperature1(C)"])] for trialdata in Angsdf[:2]]
# Bi2Te3data i from 0 to 1: 40, 50, 60, 70, 80, 90
# Bi2Te3data j from 0 to 2: time, heat source T7, respond T8
Bi2Te3data = [[np.array(trialdata[0]["Time(s)"]), np.array(trialdata[0]["Temperature7(C)"]), np.array(trialdata[1]["Temperature8(C)"])] for trialdata in Angsdf[2:]]
# Reducing data points by time
Sireddata = [0 for _ in range(2)]
Bi2Te3reddata = [0 for _ in range(6)]
for i in range(2):
    n = len(Sidata[i][0])
    l, r = int(0.2 * n), int(0.9 * n)
    Sireddata[i] = [np.copy(arr[l:r]) for arr in Sidata[i]]
for i in range(6):
    n = len(Bi2Te3data[i][0])
    l, r = int(0.2 * n), int(0.9 * n)
    Bi2Te3reddata[i] = [np.copy(arr[l:r]) for arr in Bi2Te3data[i]]

## Raw Data Plots  ##

Sifig, Siaxs = plt.subplots(1, 2, figsize = (12, 5))
Siredfig, Siredaxs = plt.subplots(1, 2, figsize = (12, 5))
Bi2Te3fig, Bi2Te3axs = plt.subplots(3, 2, figsize = (12, 12))
Bi2Te3redfig, Bi2Te3redaxs = plt.subplots(3, 2, figsize = (12, 12))
plt.subplots_adjust(left = 0.1, bottom = 0.05, top = 0.94, right = 0.9, wspace = 0.3, hspace = 0.25)
for i in range(2):
    Siaxs[i].plot(Sidata[i][0], Sidata[i][1], color = 'blue', label = f"Si 30s Trial{Sitrial[i]}-T2")
    Siaxs[i].plot(Sidata[i][0], Sidata[i][2], color = 'yellow', label = f"Si 30s Trial{Sitrial[i]}-T1")
    Siaxs[i].set_xlabel("Time (s)")
    Siaxs[i].set_ylabel("Temperature ($^\circ C$)")
    Siaxs[i].grid()
    Siaxs[i].legend()
    Siaxs[i].set_title(f"Si Trial{Sitrial[i]}")
    Siredaxs[i].plot(Sireddata[i][0], Sireddata[i][1], color = 'blue', label = f"Si Reduced 30s Trial{Sitrial[i]}-T2")
    Siredaxs[i].plot(Sireddata[i][0], Sireddata[i][2], color = 'yellow', label = f"Si Reduced 30s Trial{Sitrial[i]}-T1")
    Siredaxs[i].set_xlabel("Time (s)")
    Siredaxs[i].set_ylabel("Temperature ($^\circ C$)")
    Siredaxs[i].grid()
    Siredaxs[i].legend()
    Siredaxs[i].set_title(f"Si Reduced Trial{Sitrial[i]}")
for i in range(3):
    for j in range(2):
        index = 2 * i + j
        Bi2Te3axs[i][j].plot(Bi2Te3data[index][0], Bi2Te3data[index][1], color = 'blue', label = f"Bi2Te3 {Bi2Te3time[index]}s T7")
        Bi2Te3axs[i][j].plot(Bi2Te3data[index][0], Bi2Te3data[index][2], color = 'yellow', label = f"Bi2Te3 {Bi2Te3time[index]}s T8")
        Bi2Te3axs[i][j].set_xlabel("Time (s)")
        Bi2Te3axs[i][j].set_ylabel("Temperature ($^\circ C$)")
        Bi2Te3axs[i][j].grid()
        Bi2Te3axs[i][j].legend()
        Bi2Te3axs[i][j].set_title(f"Bi2Te3 {Bi2Te3time[index]}s")
        Bi2Te3redaxs[i][j].plot(Bi2Te3reddata[index][0], Bi2Te3reddata[index][1], color = 'blue', label = f"Bi2Te3 Reduced {Bi2Te3time[index]}s T7")
        Bi2Te3redaxs[i][j].plot(Bi2Te3reddata[index][0], Bi2Te3reddata[index][2], color = 'yellow', label = f"Bi2Te3 Reduced {Bi2Te3time[index]}s T8")
        Bi2Te3redaxs[i][j].set_xlabel("Time (s)")
        Bi2Te3redaxs[i][j].set_ylabel("Temperature ($^\circ C$)")
        Bi2Te3redaxs[i][j].grid()
        Bi2Te3redaxs[i][j].legend()
        Bi2Te3redaxs[i][j].set_title(f"Bi2Te3 Reduced {Bi2Te3time[index]}s")
Sifig.suptitle("Si Temperature to Time Graphs")
Sifig.savefig("../pics/Si-T1T2.png")
Siredfig.suptitle("Si Temperature to Time (Reduced) Graphs")
Siredfig.savefig("../pics/Si-Reduced-T1T2.png")
Bi2Te3fig.suptitle("Bi2Te3 Temperature to Time Graphs")
Bi2Te3fig.savefig("../pics/Bi2Te3-T7T8.png")
Bi2Te3redfig.suptitle("Bi2Te3 Temperature to Time (Reduced) Graphs")
Bi2Te3redfig.savefig("../pics/Bi2Te3-Reduced-T7T8.png")
plt.show()

## FFT ##

# Use reduced data to perform FFT

# Si FFT Figures
Sifftampfig, Sifftampaxs = plt.subplots(1, 2, figsize = (12, 5))
plt.subplots_adjust(left = 0.1, bottom = 0.1, top = 0.9, right = 0.9, wspace = 0.2, hspace = 0.25)
Sifftphasefig, Sifftphaseaxs = plt.subplots(1, 2, figsize = (12, 5))
plt.subplots_adjust(left = 0.1, bottom = 0.1, top = 0.9, right = 0.9, wspace = 0.2, hspace = 0.25)
# Siffisig[i][j][n] Amplitude spectrum of Trial i, Tj
# i: Trial 1, 2
# j: T2, T1
Sifftsig = [[0 for i in range(2)] for j in range(2)]
# Sisigfreq[i][j][n] Frequency array of Trial i, Tj
Sisigfreq = [[0 for i in range(2)] for j in range(2)]
Siamp = [[0 for i in range(2)] for j in range(2)]
Siphase = [[0 for i in range(2)] for j in range(2)]
# Sampling time interval
dt = 0.5
for i in range(2):
    # Heat source and response signals
    signal = [np.copy(Sireddata[i][1]), np.copy(Sireddata[i][2])]
    sigsize = len(signal[0])
    print(f"Si Trial {Sitrial[i]} Frequency array length: {sigsize}.\n")
    Sifftsig[i] = [fft(sig) for sig in signal]
    Sisigfreq[i] = [fftfreq(len(sig), dt) for sig in Sifftsig[i]]
    # Positive frequency indices
    posind = [freq > 0 for freq in Sisigfreq[i]]
    # Spectrum frequency array
    Sisigfreq[i] = [Sisigfreq[i][j][posind[j]] for j in range(2)]
    # Spectrum signal exponential form
    Sifftsig[i] = [Sifftsig[i][j][posind[j]] for j in range(2)]
    # Amplitude array
    Siamp[i] = [np.abs(sig) * 2 / sigsize for sig in Sifftsig[i]]
    # Phase array
    Siphase[i] = [np.angle(sig) for sig in Sifftsig[i]]
    for j in range(2):
        print(f"Si Trial {Sitrial[i]} T{endpts[j][0]} FFT Amplitude Array (First 30 Terms)")
        print(Siamp[i][j][:30])
        print(f"Si Trial {Sitrial[i]} T{endpts[j][0]} FFT Frequency Sample Array (First 30 Terms)")
        print(Sisigfreq[i][j][:30])
        print(f"Si Trial {Sitrial[i]} T{endpts[j][0]} FFT Phase Array (First 30 Terms)")
        print(Siphase[i][j][:30])
        print()
    Sifftampaxs[i].plot(Sisigfreq[i][0], Siamp[i][0], color = 'blue', label = "Si T2 Amplitude")
    Sifftampaxs[i].plot(Sisigfreq[i][0], Siamp[i][1], color = 'yellow', label = "Si T1 Amplitude")
    Sifftampaxs[i].set_xlabel("Frequency (Hz)")
    Sifftampaxs[i].set_ylabel("Temperature Amplitude ($^\circ C$)")
    Sifftampaxs[i].grid()
    Sifftampaxs[i].legend()
    Sifftampaxs[i].set_title(f"Si Trial {Sitrial[i]} Amplitude Spectrum")
    Sifftphaseaxs[i].plot(Sisigfreq[i][0], Siphase[i][0], color = 'blue', label = "Si T2 Phase")
    Sifftphaseaxs[i].plot(Sisigfreq[i][0], Siphase[i][1], color = 'yellow', label = "Si T1 Phase")
    Sifftphaseaxs[i].set_xlabel("Frequency (Hz)")
    Sifftphaseaxs[i].set_ylabel("Phase (rad)")
    Sifftphaseaxs[i].grid()
    Sifftphaseaxs[i].legend()
    Sifftphaseaxs[i].set_title(f"Si Trial {Sitrial[i]} Phase Spectrum")
# Saving figures
Sifftampfig.suptitle("Si Amplitude Spectra")
Sifftampfig.savefig("../pics/Si_FFT_Amplitude_Spectra.png")
Sifftphasefig.suptitle("Si Phase Spectra")
Sifftphasefig.savefig("../pics/Si_FFT_Phase_Spectra.png")
plt.show()

## Spectral Analysis

# Si Theoretical Peak Frequency
Sithpeakfreq = 1 / 30
# Heat capacity (J/(kg K))
Sicap = 700 
# Density (kg/m^3)
Sirho = 2330
# Distance between T1, T2 (m)
Sidx = uct.ufloat(30, 1) / 1000
# Si Trial i Tj amplitude peak frequency array index
# i: 1, 2, j: 2, 1
# !!1-base!! not 0-base
Siamppeakind = [[8, 7], [13, 13]]
# Si amplitude peak frequencies  with uncertainties
Siamppeakfreq = [[0, 0], [0, 0]]
# Si Trial i Tj phase peak frequency array index
# i: 1, 2, j: 2, 1
# !!1-base!! not 0-base
Siphasepeakind = [[7, 8], [13, 13]]
# Time difference of T1, T2 of Trial i (seconds)
Sitimediff = [0, 0]
# Si thermal conductivity (W/mK)
Sithmcdt = [0, 0]
for i in range(2):
    freqstep = Sisigfreq[i][0][0]
    for j in range(2):
        print(f"Si trial {Sitrial[i]} index of T{endpts[j][0]} amplitude peak in frequency array: {Siamppeakind[i][j]}.")
        freqpeak = Sisigfreq[i][j][Siamppeakind[i][j] - 1]
        Siamppeakfreq[i][j] = uct.ufloat(freqpeak, freqstep / 2)
        print(f"Si trial {Sitrial[i]} T{endpts[j][0]} amplitude peak frequency: {ufloat_print_format(Siamppeakfreq[i][j])} Hz.")
    # Phase peaks of T2 and T1
    phase = [Siphase[i][j][Siphasepeakind[i][j] - 1] for j in range(2)]
    # Calculating phase difference, response must be ahead of source
    phasediff = phase[1] - phase[0] if phase[1] - phase[0] >= 0 else phase[1] - phase[0] + 2 * np.pi
    for j in range(2):
        print(f"Si trial {Sitrial[i]} T{endpts[j][0]} phase peak: {phase[j]:.2f} (rad).")
    print(f"Si trial {Sitrial[i]} theoretical peak frequency: {Sithpeakfreq:.4f} Hz.")
    Sitimediff[i] = phasediff / (2 * np.pi * Sithpeakfreq)
    print(f"Si trial {Sitrial[i]} time difference: {Sitimediff[i]:.2f} s.")
    # Amplitude peaks of T2 and T1
    amp = [Siamp[i][j][Siamppeakind[i][j] - 1] for j in range(2)]
    # Thermal Conductivity Formula
    Sithmcdt[i] = Sicap * Sirho * (Sidx ** 2) / (2 * Sitimediff[i] * np.log(amp[0] / amp[1]))
    print(f"Si trial {Sitrial[i]} thermal conductivity: {Sithmcdt[i]} W/mK.")
print()

## Bi2Te3 FFT
Bi2Te3fftampfig, Bi2Te3fftampaxs = plt.subplots(3, 2, figsize = (12, 12))
plt.subplots_adjust(left = 0.1, bottom = 0.05, top = 0.94, right = 0.9, wspace = 0.3, hspace = 0.25)
# Bi2Te3ffisig[i][j][n] Amplitude spectrum of Time 40 + 10 * i sec, T7 or T8, resp. j
# i from 0 to 5: 40, 50, 60, 70, 80, 90
# j from 0 to 1: T7, T8
Bi2Te3fftsig = [[0 for i in range(2)] for j in range(6)]
Bi2Te3sigfreq = [[0 for i in range(2)] for j in range(6)]
Bi2Te3amp = [[0 for i in range(2)] for j in range(6)]
Bi2Te3phase = [[0 for i in range(2)] for j in range(6)]
dt = 0.5
for i in range(3):
    for j in range(2):
        index = i * 2 + j
        signal = [np.copy(Bi2Te3reddata[index][1]), np.copy(Bi2Te3reddata[index][2])]
        sigsize = len(signal[0])
        print(f"Bi2Te3 {Bi2Te3time[index]}s Frequency array length: {sigsize}.\n")
        Bi2Te3fftsig[index] = [fft(sig) for sig in signal]
        Bi2Te3sigfreq[index] = [fftfreq(len(sig), dt) for sig in Bi2Te3fftsig[index]]
        # Positive frequency indices
        posind = [freq > 0 for freq in Bi2Te3sigfreq[index]]
        Bi2Te3sigfreq[index] = [Bi2Te3sigfreq[index][k][posind[k]] for k in range(2)]
        Bi2Te3fftsig[index] = [Bi2Te3fftsig[index][k][posind[k]] for k in range(2)]
        # Amplitude array
        Bi2Te3amp[index] = [np.abs(sig) * 2 / sigsize for sig in Bi2Te3fftsig[index]] 
        # Phase array
        Bi2Te3phase[index] = [np.angle(sig) for sig in Bi2Te3fftsig[index]]
        for k in range(2):
            print(f"Bi2Te3 {Bi2Te3time[index]}s T{endpts[k][1]} FFT Amplitude Array (First 30 Terms)")
            print(Bi2Te3amp[index][k][:30])
            print(f"Bi2Te3 {Bi2Te3time[index]}s T{endpts[k][1]} FFT Frequency Sample Array (First 30 Terms)")
            print(Bi2Te3sigfreq[index][k][:30])
            print(f"Bi2Te3 {Bi2Te3time[index]}s T{endpts[k][1]} FFT Phase Array (First 30 Terms)")
            print(Bi2Te3phase[index][k][:30])
            print()
        Bi2Te3fftampaxs[i][j].plot(Bi2Te3sigfreq[index][0], Bi2Te3amp[index][0], color = 'blue', label = "Bi2Te3 T7 FFT")
        Bi2Te3fftampaxs[i][j].plot(Bi2Te3sigfreq[index][0], Bi2Te3amp[index][1], color = 'yellow', label = "Bi2Te3 T8 FFT")
        Bi2Te3fftampaxs[i][j].set_xlabel("Frequency (Hz)")
        Bi2Te3fftampaxs[i][j].set_ylabel("Temperature Amplitude ($^\circ C$)")
        Bi2Te3fftampaxs[i][j].grid()
        Bi2Te3fftampaxs[i][j].legend()
        Bi2Te3fftampaxs[i][j].set_title(f"Bi2Te3 {Bi2Te3time[index]}s Amplitude Spectrum")
Bi2Te3fftampfig.suptitle("Bi2Te3 Amplitude Spectra")
Bi2Te3fftampfig.savefig("../pics/Bi2Te3_FFT_Amplitude_Spectra.png")

## Focusing on 80s, 90s cases
Bi2Te3fftphasefig, Bi2Te3fftphaseaxs = plt.subplots(1, 2, figsize = (12, 5))
plt.subplots_adjust(left = 0.1, bottom = 0.1, top = 0.9, right = 0.9, wspace = 0.2, hspace = 0.25)
for i in range(2):
    Bi2Te3fftphaseaxs[i].plot(Bi2Te3sigfreq[i + 4][0], Bi2Te3phase[i + 4][0], color = 'blue', label = "Bi2Te3 T7 Phase")
    Bi2Te3fftphaseaxs[i].plot(Bi2Te3sigfreq[i + 4][0], Bi2Te3phase[i + 4][1], color = 'yellow', label = "Bi2Te3 T8 Phase")
    Bi2Te3fftphaseaxs[i].set_xlabel("Frequency (Hz)")
    Bi2Te3fftphaseaxs[i].set_ylabel("Phase (rad)")
    Bi2Te3fftphaseaxs[i].grid()
    Bi2Te3fftphaseaxs[i].legend()
    Bi2Te3fftphaseaxs[i].set_title(f"Bi2Te3 {Bi2Te3time[i + 4]}s Phase Spectrum")
Bi2Te3fftphasefig.suptitle("Bi2Te3 Phase Spectra")
Bi2Te3fftphasefig.savefig("../pics/Bi2Te3_FFT_Phase_Spectra.png")
plt.show()

## Spectral Analysis

Bi2Te3thpeakfreq = [1 / 80, 1 / 90]
# Heat capacity
Bi2Te3cap = 126.19 / (800.761 / 1000)
# Density
Bi2Te3rho = 6966
# Distance between T7, T8
Bi2Te3dx = uct.ufloat(30, 1) / 1000
# Bi2Te3 80s, 90s Amplitude Peak Frequency Array Index
Bi2Te3amppeakind = [[6, 6], [5, 5]]
Bi2Te3freqpeak = [[0, 0], [0, 0]]
# Bi2Te3 Time i Tj Phase Peak Frequency Array Index
# i: 80s, 90s, j: 7, 8
# !!1-base!! not 0-base
Bi2Te3phasepeakind = [[5, 5], [5, 5]]
# Time difference of T1, T2 of Trial i
Bi2Te3timediff = [0, 0]
# Bi2Te3 thermal conductivity (W / m K)
Bi2Te3thmcdt = [0, 0]
for i in range(2):
    freqstep = Bi2Te3sigfreq[i + 4][0][0]
    for j in range(2):
        freqpeak = Bi2Te3sigfreq[i + 4][j][Bi2Te3amppeakind[i][j] - 1]
        Bi2Te3freqpeak[i][j] = uct.ufloat(freqpeak, freqstep / 2)
        print(f"Bi2Te3 {Bi2Te3time[i + 4]}s index of T{endpts[j][1]} amplitude peak in frequency array: {Bi2Te3amppeakind[i][j]}.")
        print(f"Bi2Te3 {Bi2Te3time[i + 4]}s T{endpts[j][i]} amplitude peak frequency: {ufloat_print_format(Bi2Te3freqpeak[i][j])} Hz.")
    phase = [Bi2Te3phase[i + 4][j][Bi2Te3phasepeakind[i][j] - 1] for j in range(2)]
    phasediff = phase[1] - phase[0] if phase[1] - phase[0] >= 0 else phase[1] - phase[0] + 2 * np.pi
    for j in range(2):
        print(f"Bi2Te3 {Bi2Te3time[i + 4]}s T{endpts[j][1]} phase peak: {phase[j]:.2f} (rad).")
    print(f"Bi2Te3 {Bi2Te3time[i + 4]}s theoretical peak frequency: {Bi2Te3thpeakfreq[i]:.4f} Hz.")
    Bi2Te3timediff[i] = phasediff / (2 * np.pi * Bi2Te3thpeakfreq[i])
    print(f"Bi2Te3 {Bi2Te3time[i + 4]}s time difference: {Bi2Te3timediff[i]:.2f} s.")
    amp = [Bi2Te3amp[i + 4][j][Bi2Te3amppeakind[i][j] - 1] for j in range(2)]
    Bi2Te3thmcdt[i] = Bi2Te3cap * Bi2Te3rho * (Bi2Te3dx ** 2) / (2 * Bi2Te3timediff[i] * np.log(amp[0] / amp[1]))
    # Thermal Conductivity
    print(f"Bi2Te3 {Bi2Te3time[i + 4]}s thermal conductivity: {Bi2Te3thmcdt[i]} W/mK.")
