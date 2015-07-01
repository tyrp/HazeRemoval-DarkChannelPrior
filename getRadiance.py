import numpy as np

def getRadiance_(A=None, im=None, tMat=None, *args, **kwargs):

    t0 = 0.1
    J = np.zeros(im.shape)
    for ind in xrange(0,3):
        J[:, :, ind] = A[ind] + (im[:, :, ind] - A[ind]) / np.maximum(tMat, t0)

    J = J / np.amax(J)
    return J