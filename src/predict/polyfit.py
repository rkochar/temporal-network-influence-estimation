import numpy as np


def polyfit(data, new_points):
    fit = np.polyfit(data[:,0], data[:,1] ,1) #The use of 1 signifies a linear fit.

    line = np.poly1d(fit)
    new_points = np.arange(new_points) + (data[-1, 0] + 1)

    return np.cumsum(line(new_points))