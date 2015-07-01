import numpy as np
import math, cv2
from scipy import misc
import matplotlib.pyplot as plt
from scipy.sparse import *
import scipy.ndimage as ndimage
import scipy.sparse.linalg as spla
import numpy.matlib

def readIm(im=None, ext=None, *args, **kwargs):
    return cv2.imread(im, 1)

def getDarkChannel(im=None, *args, **kwargs):
    height, width, __ = im.shape
    numWindowPixels = 15
    padding = math.ceil(numWindowPixels / 2.0)
    J = np.zeros((height,width))
    paddedImage = np.pad(im, (padding, padding), 'constant', constant_values=(np.inf, np.inf))
    for j in xrange(0, height):
        for i in xrange(0, width):
            window = paddedImage[j : j + numWindowPixels - 1, i : i + numWindowPixels - 1, :]
            J[j, i] = np.amin(window)
    return J

def getAtmLight(im=None, JDark=None, *args, **kwargs):
    height, width, __ = im.shape
    totalPixels = width * height
    
    ImVec = np.reshape(im, (totalPixels, 3))
    indices = np.argsort(np.reshape(JDark, (totalPixels, 1)), axis=0).flatten()

    topPixels = math.floor(totalPixels / 1000.0)
    indices = indices[-topPixels:]
    tempAtm = np.zeros((1,3))

    for ind in xrange(0, int(topPixels)):
        tempAtm = tempAtm + ImVec[indices[ind], :]

    A = tempAtm / topPixels

    return A.flatten()

def getTransmission(im=None, A=None, *args, **kwargs):
    omega = 0.95
    newImage = np.zeros(im.shape)
    for ind in xrange(0,3):
        newImage[:, :, ind] = im[:, :, ind] / A[ind]

    return 1 - omega * getDarkChannel(newImage)

def getRadiance(atmLight=None, im=None, transmission=None, *args, **kwargs):
    t0 = 0.1
    J = np.zeros(im.shape)
    for ind in xrange(0,3):
        J[:, :, ind] = atmLight[ind] + (im[:, :, ind] - atmLight[ind]) / np.maximum(transmission, t0)

    return J / np.amax(J)

neighbors = []
neighbor_count = 0;

def performSoftMatting(im=None, transmission=None, *args, **kwargs):
    global neighbors

    width, height, depth = im.shape
    windowRadius = 1
    numWindowPixels = 9
    epsilon = 10 ** - 8
    _lambda = 10 ** - 4

    totalPixels = numWindowPixels ** 2
    windowWidth = 3
    windowIndicies = np.reshape(xrange(1, width * height + 1), (width, height), order='F')
    totalElements = totalPixels * (width - 2) * (height - 2)
    xIndicies = np.ones((1, totalElements))
    yIndicies = np.ones((1, totalElements))
    laplacian = np.zeros((1, totalElements))
    count = 0

    neighbors = np.empty((width * height, numWindowPixels))

    footprint = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]])

    ndimage.generic_filter(windowIndicies, getWindow, footprint=footprint)

    U = epsilon / numWindowPixels * identity(windowWidth)

    for i in xrange(windowRadius, height - windowRadius):
        for j in xrange(windowRadius, width - windowRadius):
            window = im[j - windowRadius: j + windowRadius + 1, i - windowRadius : i + windowRadius + 1, :]

            reshapedWindow = np.reshape(window, (numWindowPixels, 3), order='F')

            diffFromMean = reshapedWindow.T - np.tile(np.mean(reshapedWindow, axis=0).T, (numWindowPixels, 1)).T

            window_covariance = np.dot(diffFromMean, diffFromMean.T) / numWindowPixels

            entry = identity(numWindowPixels) - (1 + np.dot(np.dot(diffFromMean.T, np.linalg.inv(window_covariance + U)), diffFromMean)) / float(numWindowPixels)

            temp = count * totalPixels
            temp2 = count * totalPixels + totalPixels

            newIndicies = np.tile(np.reshape(np.reshape(neighbors[height * j + i], (windowWidth, windowWidth)), (1, numWindowPixels), order='F'), (numWindowPixels, 1))

            xIndicies[0][temp : temp2] = newIndicies.flatten(1)
            yIndicies[0][temp : temp2] = (newIndicies.T).flatten(1)
            laplacian[0][temp : temp2] = entry.flatten(1)
            count += 1

    L = csc_matrix((laplacian.flatten(), (xIndicies.flatten(), yIndicies.flatten())))
    tBar = np.append(np.reshape(transmission.T, (width * height, 1)), [0])

    T = spla.spsolve(L + _lambda * identity(L.shape[0]), tBar * _lambda) 
    return np.reshape(np.delete(T, len(T) - 1), transmission.shape, order='F')

def getWindow(values):
    global neighbors, neighbor_count
    neighbors[neighbor_count] = np.reshape([values], (1,9))
    neighbor_count += 1
    return 0