import math
import numpy as np

def atmLight_(im=None,JDark=None,*args,**kwargs):

    height, width, __ = im.shape
    imsize = width * height
    numpx = math.floor(imsize / 1000.0)
    
    JDarkVec = np.reshape(JDark, (imsize, 1))
    ImVec = np.reshape(im, (imsize, 3))
    indices = np.argsort(JDarkVec, axis=0)
    indices = indices.flatten()
    indices = indices[-numpx:]
    atmSum = np.zeros((1,3))

    for ind in xrange(0, int(numpx)):
        atmSum = atmSum + ImVec[indices[ind], :]

    A = atmSum / numpx

    return A.flatten()