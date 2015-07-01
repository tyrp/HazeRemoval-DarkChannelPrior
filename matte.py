from scipy.sparse import *
import scipy.ndimage as ndimage
import scipy.sparse.linalg as spla
import numpy as np
import numpy.matlib

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

    U = epsilon / numWindowPixels * identity(3)

    for i in xrange(0 + windowRadius, height - windowRadius):
        for j in xrange(0 + windowRadius, width - windowRadius):
            print 'i', i
            print 'j', j
            window = im[j - windowRadius: j + windowRadius + 1, i - windowRadius : i + windowRadius + 1, :]

            reshapedWindow = np.reshape(window, (numWindowPixels, 3), order='F')

            diffFromMean = reshapedWindow.T - np.tile(np.mean(reshapedWindow, axis=0).T, (numWindowPixels, 1)).T

            window_covariance = np.dot(diffFromMean, diffFromMean.T) / numWindowPixels

            entry = identity(numWindowPixels) - (1 + np.dot(np.dot(diffFromMean.T, np.linalg.inv(window_covariance + U)), diffFromMean)) / float(numWindowPixels)

            temp = count * totalPixels
            temp2 = count * totalPixels + totalPixels

            iterationNeighbors = np.reshape(np.reshape(neighbors[height * j + i], (3, 3)), (1, numWindowPixels), order='F')

            x = np.tile(iterationNeighbors, (numWindowPixels, 1))
            y = (x.T).flatten(1)

            xIndicies[0][temp : temp2] = x.flatten(1)
            yIndicies[0][temp : temp2] = y
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