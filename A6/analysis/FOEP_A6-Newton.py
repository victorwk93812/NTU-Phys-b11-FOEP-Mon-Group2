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

### Newton's Cooling Law ###

numlst = [2, 3, 6, 7]
# List of materials
matlst = ["Si", "Brass", "Al", "Bi2Te3"]
# Thermal conductivity (W/mK) list
thmcnd = [148, 105, 237, 1.20]
ds = [f"../data/Newton_T{x}.csv" for x in numlst]
df = [pd.read_csv(datsrc) for datsrc in ds]
time = [np.array(df["Time(s)"]) for df in df]
temp = [np.array(df[f"Temperature{num}(C)"]) for num, df in zip(numlst, df)]

# Newton original plots
fig, axs= plt.subplots()
for i in range(4):
    axs.plot(time[i], temp[i], label = f"Newton T{numlst[i]}")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Temperature $(^\circ C)$")
axs.set_title("Materials Temperature-Time Graph")
axs.grid()
axs.legend()
fig.suptitle("Newton's Cooling Law")
fig.savefig("../pics/Newton_Cooling.png")
# plt.show()
# Index left and right bounds for time and temperature arrays
indbnd = [[221, 542], [221, 542], [221, 542], [261, 542]]
time = [time[i][indbnd[i][0]:indbnd[i][1]] for i in range(4)]
temp = [temp[i][indbnd[i][0]:indbnd[i][1]] for i in range(4)]
fig, axs= plt.subplots()
for i in range(4):
    axs.plot(time[i], temp[i], label = f"{matlst[i]} (T{numlst[i]})")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Temperature $(^\circ C)$")
axs.set_title("Materials Cooling")
axs.grid()
axs.legend()
fig.suptitle("Newton's Cooling Law")
fig.savefig("../pics/Newton_Cooling-Reduced.png")
# plt.show()

def exp_curve(t, A, B, tau):
    return A + B * np.exp(- ((t - t[0]) / tau))

# Initial guess of A, B, tau
init_guess = [25., 3., 1.]
popt = [curve_fit(exp_curve, time[i], temp[i])[0] for i in range(4)]
A = [popt[i][0] for i in range(4)]
B = [popt[i][1] for i in range(4)]
tau = [popt[i][2] for i in range(4)]

# Curve fit plots
tempfitcur = [exp_curve(time[i], *(popt[i])) for i in range(4)]
fig, axs= plt.subplots()
for i in range(4):
    axs.scatter(time[i], temp[i], s = 5, label = f"{matlst[i]} (T{numlst[i]})")
    axs.plot(time[i], tempfitcur[i], label = f"{matlst[i]} (T{numlst[i]}) Fitted Curve")
axs.set_xlabel("Time (s)")
axs.set_ylabel("Temperature $(^\circ C)$")
axs.set_title("Fitting Materials Cooling Curve")
axs.grid()
axs.legend()
fig.suptitle("Newton's Cooling Law")
fig.savefig("../pics/Newton_Cooling-Reduced-Fitted.png")
# plt.show()

for i in range(4):
    print(f"Time constant of {matlst[i]} is {popt[i][2]:.2f} s.\n")
