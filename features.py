# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:08:49 2016

@author: cs390mb

This file is used for extracting features over windows of tri-axial accelerometer
data. We recommend using helper functions like _compute_mean_features(window) to
extract individual features.

As a side note, the underscore at the beginning of a function is a Python
convention indicating that the function has private access (although in reality
it is still publicly accessible).

"""

import numpy as np
import math

lastLatitude = 0
lastLongtitude = 0

def compute_crossing(window):
    list1 = np.sign(window[1:])
    list2 = np.sign(window[:-1])
    combineList = ((list1 - list2)!=0)
    return np.sum(combineList)

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window.
    """
    return np.mean(window[:,:3], axis=0)

def _compute_variance_features(window):
    return np.var(window[:,:3],axis=0)

def _compute_min_features(window):
    return np.amin(window[:,:3],axis=0)

def _compute_max_features(window):
    return np.amax(window[:,:3],axis=0)

def _compute_median_features(window):
    return np.median(window[:,:3],axis=0)

def _compute_zero_crossing_features(window):
    xArray = window[:,0]
    yArray = window[:,1]
    zArray = window[:,2]
    xCrossingRate = compute_crossing(xArray)
    yCrossingRate = compute_crossing(yArray)
    zCrossingRate = compute_crossing(zArray)
    return np.array([[xCrossingRate],[yCrossingRate],[zCrossingRate]])

def _compute_mean_crossing_features(window):
    meanArray = _compute_mean_features(window)
    xArray = window[:,0]
    yArray = window[:,1]
    zArray = window[:,2]
    xCrossingRate = compute_crossing(xArray-meanArray[0])
    yCrossingRate = compute_crossing(yArray-meanArray[1])
    zCrossingRate = compute_crossing(zArray-meanArray[2])
    return np.array([[xCrossingRate],[yCrossingRate],[zCrossingRate]])

def _compute_mean_magnitude_singal(window):
    return np.mean(np.sqrt(np.sum(np.square(window[:,:3]),axis=1)))

def _compute_FFT_features(window):
    n_freq = 32
    freq = np.fft.fftfreq(n_freq)
    spX = sp = np.fft.fft(window[:,0], n=n_freq).astype(float)
    spY = sp = np.fft.fft(window[:,1], n=n_freq).astype(float)
    spZ = sp = np.fft.fft(window[:,2], n=n_freq).astype(float)
    dominantX = freq[spX.argmax()]
    dominantY = freq[spY.argmax()]
    dominantZ = freq[spZ.argmax()]
    return np.array([[dominantX],[dominantY],[dominantZ]])

def _compute_entropy(window):
    firstArray = (np.histogram(window)[0]).astype(float)
    filterArray = [result for result in firstArray if result > 0]
    logArray = [-(value)*math.log(value) for value in filterArray]
    return np.sum(logArray)

def compute_peak(window):
    counter = 0
    status = "increasing"
    for i in range(len(window)-1):
        if(status == "increasing"):
            if(i+1<i):
                counter += 1
            else:
                status = "decreasing"
        else:
            if(i+1>i):
                status = "increasing"
    return counter

def _compute_peak_features(window):
    xArray = window[:,0]
    yArray = window[:,1]
    zArray = window[:,2]
    xPeak = compute_peak(xArray)
    yPeak = compute_peak(yArray)
    zPeak = compute_peak(zArray)
    return np.array([[xPeak],[yPeak],[zPeak]])

def _compute_location_distance(window):
    global lastLatitude,lastLongtitude
    latitude = np.median(window[:,4])
    longtitude = np.median(window[:,5])
    if (latitude==0 and lastLatitude==0):
        lastLatitude = latitude
        lastLongtitude = longtitude

    diffLat = math.radians(latitude - lastLatitude)
    diffLong = math.radians(longtitude - lastLongtitude)
    haversinLng = (1 - math.cos(diffLong))/2
    haversinLat = (1 - math.cos(diffLat))/2
    haversinCentralAngle = haversinLat + math.cos(math.radians(latitude))
    math.cos(math.radians(lastLongtitude)) * haversinLng
    d = 2 * math.atan2(math.sqrt(haversinCentralAngle),math.sqrt(1-haversinCentralAngle))
    lastLatitude = latitude
    lastLongtitude = longtitude
    return 6371000 * d


def _compute_heartRate(window):
    return np.mean(window[:,3])



def extract_features(window):
    """
    Here is where you will extract your features from the data over
    the given window. We have given you an example of computing
    the mean and appending it to the feature matrix X.

    Make sure that X is an N x d matrix, where N is the number
    of data points and d is the number of features.

    """



    x = []

    x = np.append(x, _compute_mean_features(window))
    x = np.append(x,_compute_variance_features(window))
    x = np.append(x,_compute_min_features(window))
    x = np.append(x,_compute_max_features(window))
    x = np.append(x,_compute_median_features(window))
    x = np.append(x,_compute_zero_crossing_features(window))
    x = np.append(x,_compute_mean_crossing_features(window))
    x = np.append(x,_compute_FFT_features(window))
    x = np.append(x,_compute_mean_magnitude_singal(window))
    x = np.append(x,_compute_entropy(window))
    x = np.append(x,_compute_peak_features(window))
    x = np.append(x,_compute_location_distance(window))
    x = np.append(x,_compute_heartRate(window))

    return x
