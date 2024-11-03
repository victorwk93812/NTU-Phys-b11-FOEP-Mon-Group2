import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import uncertainties as uct
from uncertainties import unumpy as unp

def adduct(arr, uarr):
    """Given array and corresponding uncertainty array, return unumpy array"""
    n = len(arr)
    result = np.array([])
    for i in range(n):
        result = np.append(result, uct.ufloat(arr[i], uarr[i]))
    return result

def uctarravg(arr):
    """Given unumpy arrat, return average"""
    n = len(arr)
    result = uct.ufloat(0, 0)
    for i in range(n):
        result += arr[i]    
    result /= n
    return result

def linreg(xarr, yarr):
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

data1 = pd.read_csv("../data/FOEP_A2-1-2.csv")
data2 = pd.read_csv("../data/FOEP_A2-2-2.csv")

### debug ###
# print(data1, "\n", data2)
# print(data1.columns)
### debug ###

### Week1 ###
print("\n20min sample\n")

ICD1 = np.array(data1["ICD(mA)"])
uICD1 = np.array(data1["uICD(mA)"])
uctICD1 = adduct(ICD1, uICD1)

VBA1 = np.array(data1["VBA(mV)"])
uVBA1 = np.array(data1["uVBA(mV)"])
uctVBA1 = adduct(VBA1, uVBA1)

RCDBA1 = linreg(uctICD1, uctVBA1)[0]
print("RCDBA1:", repr(RCDBA1))

IBC1 = np.array(data1["IBC(mA)"])
uIBC1 = np.array(data1["uIBC(mA)"])
uctIBC1 = adduct(IBC1, uIBC1)

VDA1 = np.array(data1["VDA(mV)"])
uVDA1 = np.array(data1["uVDA(mV)"])
uctVDA1 = adduct(VDA1, uVDA1)

RBCDA1 = linreg(uctIBC1, uctVDA1)[0]
print("RBCDA1:", repr(RBCDA1))

r1 = max(RCDBA1.n, RBCDA1.n) / min(RCDBA1.n,RBCDA1.n)
f1 = 0.92

d1 = 15.2 * 20e-7
print("d1:", d1)

Rs1 = (np.pi * (RCDBA1 + RBCDA1) * f1) / (np.log(2) * 2)
rho1 = Rs1 * d1
print("Rs1:", repr(Rs1))
print("rho1:", repr(rho1))

IBD1 = np.array(data1["IBD(mA)"])
uIBD1 = np.array(data1["uIBD(mA)"])
uctIBD1 = adduct(IBD1, uIBD1)

VACp1 = np.array(data1["VAC+(mV)"])
uVACp1 = np.array(data1["uVAC+(mV)"])
uctVACp1 = adduct(VACp1, uVACp1)

VACn1 = np.array(data1["VAC-(mV)"])
uVACn1 = np.array(data1["uVAC-(mV)"])
uctVACn1 = adduct(VACn1, uVACn1)

RBDACp1 = linreg(uctIBD1, uctVACp1)[0]
RBDACn1 = linreg(uctIBD1, uctVACn1)[0]
DRBDAC1 = (RBDACp1 - RBDACn1) / 2
print("RBDAC+1:", repr(RBDACp1))
print("RBDAC-1:", repr(RBDACn1))

B = 0.37

mu1 = (10000 * DRBDAC1) / (B * Rs1)
print("mu1:", repr(mu1))

e = 1.618e-19

n1 = 1 / (e * rho1 * mu1)
print("n1:", repr(n1))

### Week 2 ###
print("\n10min sample\n")

ICD2 = np.array(data2["ICD(mA)"])
uICD2 = np.array(data2["uICD(mA)"])
uctICD2 = adduct(ICD2, uICD2)

VBA2 = np.array(data2["VBA(mV)"])
uVBA2 = np.array(data2["uVBA(mV)"])
uctVBA2 = adduct(VBA2, uVBA2)

RCDBA2 = linreg(uctICD2, uctVBA2)[0]
print("RCDBA2:", repr(RCDBA2))

IBC2 = np.array(data2["IBC(mA)"])
uIBC2 = np.array(data2["uIBC(mA)"])
uctIBC2 = adduct(IBC2, uIBC2)

VDA2 = np.array(data2["VDA(mV)"])
uVDA2 = np.array(data2["uVDA(mV)"])
uctVDA2 = adduct(VDA2, uVDA2)

RBCDA2 = linreg(uctIBC2, uctVDA2)[0]
print("RBCDA2:", repr(RBCDA2))

r2 = max(RCDBA2.n, RBCDA2.n) / min(RCDBA2.n,RBCDA2.n)
print(repr(r2))
f2 = 0.97

d2 = 15.2 * 10e-7
print("d2:", d2)

Rs2 = (np.pi * (RCDBA2 + RBCDA2) * f2) / (np.log(2) * 2)
rho2 = Rs2 * d2
print("Rs2:", repr(Rs2))
print("rho2:", repr(rho2))

IBD2 = np.array(data2["IBD(mA)"])
uIBD2 = np.array(data2["uIBD(mA)"])
uctIBD2 = adduct(IBD2, uIBD2)

VACp2 = np.array(data2["VAC+(mV)"])
uVACp2 = np.array(data2["uVAC+(mV)"])
uctVACp2 = adduct(VACp2, uVACp2)

VACn2 = np.array(data2["VAC-(mV)"])
uVACn2 = np.array(data2["uVAC-(mV)"])
uctVACn2 = adduct(VACn2, uVACn2)

RBDACp2 = linreg(uctIBD2, uctVACp2)[0]
RBDACn2 = linreg(uctIBD2, uctVACn2)[0]
DRBDAC2 = (RBDACp2 - RBDACn2) / 2
print("RBDAC+2:", repr(RBDACp2))
print("RBDAC-2:", repr(RBDACn2))

B = 0.37

mu2 = (10000 * DRBDAC2) / (B * Rs2)
print("mu2:", repr(mu2))

e = 1.618e-19

n2 = 1 / (e * rho2 * mu2)
print("n2:", repr(n2))



















