import matplotlib.pyplot as plt
import numpy as np

# Opening files
curiedata = ["../data/Group2_curie-1.txt", "../data/Group2_curie-2.txt"]
datafile = [open(curiedata[0], 'r'), open(curiedata[1], 'r')]

# Building raw data array, temperature array and resistance array
# 0: First week, 1: Second week
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

# Delete data in front with temp < 30 and at end with temp < 150
maxtemp, maxtempind = [0, 0], [0, 0]
head, tail = [0, 0], [0, 0]
for i in range(2):
    n = len(temparr[i])
    for j in range(n):
        if temparr[i][j] > maxtemp[i]:
            maxtemp[i], maxtempind[i]= temparr[i][j], j
    for j in range(maxtempind[i]):
        if temparr[i][j] > 30:
            head[i] = j
            break
    for j in range(maxtempind[i], n):
        if temparr[i][j] < 150:
            tail[i] = j
            break
    temparr[i] = temparr[i][head[i]:tail[i]]
    resarr[i] = resarr[i][head[i]:tail[i]]
    dataarr[i] = dataarr[i][head[i]:tail[i]]
datasize = [len(temparr[0]), len(temparr[1])]

print("First week datasize:", datasize[0])
print("Second week datasize:", datasize[1], '\n')

# R-T Graph
fig1, axs1 = plt.subplots(1, 2, figsize = (12, 4))
axs1[0].set_title("First Week R-T")
axs1[1].set_title("Second Week R-T")
for i in range(2):
    axs1[i].set_xlabel("Temperature (Celcius)")
    axs1[i].set_ylabel("Resistance (Omega)")

for i in range(2):
    axs1[i].plot(temparr[i], resarr[i])
fig1.savefig("../pics/R-T_Graph.png")

# dR/dT-Time Graph
fig2, axs2 =plt.subplots(1, 2, figsize = (12, 4))
axs2[0].set_title("First Week dR/dT-Time")
axs2[1].set_title("Second Week dR/dT-Time")
for i in range(2):
    axs2[i].set_xlabel("Time")
    axs2[i].set_ylabel("dR/dT")

space = 30
slopearr = [0, 0]
timeind = [0, 0]
for i in range(2):
    tmpslopearr = np.array([])
    tmptimeind = np.array([])
    for j in range(space, datasize[i] - space - 1, (2 * space + 1)):
        slopeavg = 0
        cnt = 0
        # print([j, j + space])
        for k in range(space + 1):
            tempdiff = temparr[i][j - k + space] - temparr[i][j - k]
            resdiff = resarr[i][j - k + space] - resarr[i][j - k]
            if tempdiff != 0:
                slope = resdiff / tempdiff
                cnt += 1
            slopeavg += slope
        slopeavg /= cnt
        tmpslopearr = np.append(tmpslopearr, [slopeavg])
        tmptimeind = np.append(tmptimeind, j)
    slopearr[i] = tmpslopearr
    timeind[i] = tmptimeind

# Debug slope array
# for i in range(2):
#     print(timeind[i])
#     print(slopearr[i])

for i in range(2):
    axs2[i].plot(timeind[i], slopearr[i])
fig2.savefig("../pics/dRdT-Time_Graph.png")
plt.show()

# Determining slope peaks
slopearrsize = [len(slopearr[0]), len(slopearr[1])]
windowsize = 70
halfwindowsize = windowsize // 2

firstpeaktimeind = [0, 0]
for i in range(2):
    for j in range(halfwindowsize , slopearrsize[i] - halfwindowsize - 1):
        tmpmaxslope = 0
        for k in range(windowsize):
            tmpmaxslope = max(slopearr[i][j - halfwindowsize + k], tmpmaxslope)
        if slopearr[i][j] == tmpmaxslope:
            firstpeaktimeind[i] = int(timeind[i][j])
            break
print("First week curie temperature time index on increasing temperature:", firstpeaktimeind[0])
print("Second week curie temperature time index on increasing temperature:", firstpeaktimeind[1], '\n')

lastpeaktimeind = [0, 0]
for i in range(2):
    for j in reversed(range(halfwindowsize, slopearrsize[i] - halfwindowsize - 1)):
        tmpmaxslope = 0
        for k in range(windowsize):
            tmpmaxslope = max(slopearr[i][j - halfwindowsize + k], tmpmaxslope)
        if slopearr[i][j] == tmpmaxslope:
            lastpeaktimeind[i] = int(timeind[i][j])
            break
print("First week curie temperature time index on decreasing temperature:", lastpeaktimeind[0])
print("Second week curie temperature time index on decreasing temperature:", lastpeaktimeind[1], '\n')


print("First week curie temperature on increasing temperature:", temparr[0][firstpeaktimeind[0]], "Celsius")
print("First week curie temperature on decreasing temperature:", temparr[0][lastpeaktimeind[0]], "Celsius", '\n')
print("Second week curie temperature on increasing temperature:", temparr[1][firstpeaktimeind[1]], "Celsius")
print("Second week curie temperature on decreasing temperature:", temparr[1][lastpeaktimeind[1]], "Celsius", '\n')

# Closing files
datafile[0].close()
datafile[1].close()
