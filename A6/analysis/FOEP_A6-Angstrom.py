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

# print(Angsdf[7][1].head())

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


# Angstrom Plots
Sifig, Siaxs = plt.subplots(1, 2, figsize = (10, 6))
Siredfig, Siredaxs = plt.subplots(1, 2, figsize = (10, 6))
Bi2Te3fig, Bi2Te3axs = plt.subplots(3, 2, figsize = (10, 16))
Bi2Te3redfig, Bi2Te3redaxs = plt.subplots(3, 2, figsize = (10, 16))
for i in range(2):
    Siaxs[i].plot(Sidata[i][0], Sidata[i][1], color = 'blue', label = f"Si 30s Trial{Sitrial[i]}-T2")
    Siaxs[i].plot(Sidata[i][0], Sidata[i][2], color = 'yellow', label = f"Si 30s Trial{Sitrial[i]}-T1")
    Siaxs[i].grid()
    Siaxs[i].legend()
    Siaxs[i].set_title(f"Si Trial{Sitrial[i]}")
    Siredaxs[i].plot(Sireddata[i][0], Sireddata[i][1], color = 'blue', label = f"Si Reduced 30s Trial{Sitrial[i]}-T2")
    Siredaxs[i].plot(Sireddata[i][0], Sireddata[i][2], color = 'yellow', label = f"Si Reduced 30s Trial{Sitrial[i]}-T1")
    Siredaxs[i].grid()
    Siredaxs[i].legend()
    Siredaxs[i].set_title(f"Si Reduced Trial{Sitrial[i]}")
for i in range(3):
    for j in range(2):
        index = 2 * i + j
        Bi2Te3axs[i][j].plot(Bi2Te3data[index][0], Bi2Te3data[index][1], color = 'blue', label = f"Bi2Te3 {Bi2Te3time[index]}s T7")
        Bi2Te3axs[i][j].plot(Bi2Te3data[index][0], Bi2Te3data[index][2], color = 'yellow', label = f"Bi2Te3 {Bi2Te3time[index]}s T8")
        Bi2Te3axs[i][j].grid()
        Bi2Te3axs[i][j].legend()
        Bi2Te3axs[i][j].set_title(f"Bi2Te3 {Bi2Te3time[index]}s")
        Bi2Te3redaxs[i][j].plot(Bi2Te3reddata[index][0], Bi2Te3reddata[index][1], color = 'blue', label = f"Bi2Te3 Reduced {Bi2Te3time[index]}s T7")
        Bi2Te3redaxs[i][j].plot(Bi2Te3reddata[index][0], Bi2Te3reddata[index][2], color = 'yellow', label = f"Bi2Te3 Reduced {Bi2Te3time[index]}s T8")
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

### FFT ###
# Use reduced data to perform FFT

## Si FFT
# Si FFT Figure
Sifftfig, Sifftaxs = plt.subplots(1, 2, figsize = (10, 6))
# Siffisig[i][j][n] Amplitude spectrum of Trial i, Tj
# i: Trial 1, 2
# j: T2, T1
Sifftsig = [[0 for i in range(2)] for j in range(2)]
# Sisigfreq[i][j][n] Frequency array of Trial i, Tj
Sisigfreq = [[0 for i in range(2)] for j in range(2)]
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
    Sisigfreq[i] = [Sisigfreq[i][j][posind[j]] for j in range(2)]
    Sifftsig[i] = [Sifftsig[i][j][posind[j]] for j in range(2)]
    # Amplitude array
    amp = [np.abs(sig) * 2 / sigsize for sig in Sifftsig[i]]
    phase = [np.angle(sig) for sig in Sifftsig[i]]
    for j in range(2):
        print(f"Si Trial {Sitrial[i]} T{endpts[j][0]} FFT Amplitude Array (First 30 Terms)")
        print(amp[j][:30])
        print(f"Si Trial {Sitrial[i]} T{endpts[j][0]} FFT Frequency Sample Array (First 30 Terms)")
        print(Sisigfreq[i][j][:30])
        print(f"Si Trial {Sitrial[i]} T{endpts[j][0]} FFT Phase Array (First 30 Terms)")
        print(phase[j][:30])
        print()
    # print(amp)
    # Phase array
    Sifftaxs[i].plot(Sisigfreq[i][0], amp[0], color = 'blue', label = "Si T2 FFT")
    Sifftaxs[i].plot(Sisigfreq[i][0], amp[1], color = 'yellow', label = "Si T1 FFT")
    Sifftaxs[i].grid()
    Sifftaxs[i].legend()
    Sifftaxs[i].set_title(f"Si Trial {Sitrial[i]} Amplitude Spectrum")
Sifftfig.savefig("../pics/Si_FFT_Spectrum.png")
Sifftfig.suptitle("Si Amplitude Spectra")
plt.show()
# Si Trial 1, Trial 2 Amplitude Peak Frequency Array Index
Siamppeakind = [8, 13]
Sifreqpeak = [0, 0]
for i in range(2):
    freqstep = Sisigfreq[i][0][0]
    # print(freqstep)
    freqpeak = Sisigfreq[i][0][Siamppeakind[i] - 1]
    Sifreqpeak[i] = uct.ufloat(freqpeak, freqstep / 2)
    print(f"Si trial {Sitrial[i]} index of peak in frequency array: {Siamppeakind[i]}.")
    print(f"Si trial {Sitrial[i]} peak frequency: {ufloat_print_format(Sifreqpeak[i])} Hz.")
# print("Si frequency peaks of trial 1 and 2 (Hz):")
# print(Sifreqpeak, '\n')


## Bi2Te3 FFT
Bi2Te3fftfig, Bi2Te3fftaxs = plt.subplots(3, 2, figsize = (10, 16))
# Bi2Te3ffisig[i][j][n] Amplitude spectrum of Time 40 + 10 * i sec, T7 or T8, resp. j
# i from 0 to 5: 40, 50, 60, 70, 80, 90
# j from 0 to 1: T7, T8
Bi2Te3fftsig = [[0 for i in range(2)] for j in range(6)]
# Sisigfreq[i][j][n] Frequency array of Trial i, Tj
Bi2Te3sigfreq = [[0 for i in range(2)] for j in range(6)]
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
        amp = [np.abs(sig) * 2 / sigsize for sig in Bi2Te3fftsig[index]] 
        for k in range(2):
            print(f"Bi2Te3 {Bi2Te3time[index]}s T{endpts[k][1]} FFT Amplitude Array (First 30 Terms)")
            print(amp[k][:30])
            print(f"Bi2Te3 {Bi2Te3time[index]}s T{endpts[k][1]} FFT Frequency Sample Array (First 30 Terms)")
            print(Bi2Te3sigfreq[index][k][:30])
            print(f"Bi2Te3 {Bi2Te3time[index]}s T{endpts[k][1]} FFT Phase Array (First 30 Terms)")
            print(phase[k][:30])
            print()
        # Phase array
        phase = [np.angle(sig) for sig in Bi2Te3fftsig[index]]
        Bi2Te3fftaxs[i][j].plot(Bi2Te3sigfreq[index][0], amp[0], color = 'blue', label = "Bi2Te3 T7 FFT")
        Bi2Te3fftaxs[i][j].plot(Bi2Te3sigfreq[index][0], amp[1], color = 'yellow', label = "Bi2Te3 T8 FFT")
        Bi2Te3fftaxs[i][j].grid()
        Bi2Te3fftaxs[i][j].legend()
        Bi2Te3fftaxs[i][j].set_title(f"Bi2Te3 {Bi2Te3time[index]}s Amplitude Spectrum")
Bi2Te3fftfig.savefig("../pics/Bi2Te3_FFT_Spectrum.png")
Bi2Te3fftfig.suptitle("Bi2Te3 Amplitude Spectra")
plt.show()
# Bi2Te3 80s, 90s Amplitude Peak Frequency Array Index
Bi2Te3amppeakind = [6, 5]
Bi2Te3freqpeak = [0, 0]
for i in range(2):
    freqstep = Bi2Te3sigfreq[i + 4][0][0]
    # print(freqstep)
    freqpeak = Bi2Te3sigfreq[i + 4][0][Bi2Te3amppeakind[i] - 1]
    # freqpeak = Bi2Te3amppeakind[i] * freqstep
    Bi2Te3freqpeak[i] = uct.ufloat(freqpeak, freqstep / 2)
    print(f"Bi2Te3 {Bi2Te3time[i + 4]}s index of peak in frequency array: {Bi2Te3amppeakind[i]}.")
    print(f"Bi2Te3 {Bi2Te3time[i + 4]}s peak frequency: {ufloat_print_format(Bi2Te3freqpeak[i])} Hz.")
# print("Bi2Te3 frequency peaks of 80s and 90s:")
# print(Bi2Te3freqpeak, '\n')


