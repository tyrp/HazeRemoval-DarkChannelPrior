import numpy as np
import math

def darkChannel_(im2=None, *args, **kwargs):
    height, width, __ = im2.shape
    patchSize = 15
    padSize = math.ceil(patchSize / 2.0)
    JDark = np.zeros((height,width))
    imJ = np.pad(im2, (padSize, padSize), 'constant', constant_values=(np.inf, np.inf))
    for j in xrange(0, height):
        for i in xrange(0, width):
            patch = imJ[j:(j + patchSize - 1), i:(i + patchSize - 1), :]
            JDark[j, i] = np.amin(patch[:])
    return JDark