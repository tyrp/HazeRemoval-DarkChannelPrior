from darkChannel import *

def transmissionEstimate_(im=None, A=None, *args, **kwargs):

    omega = 0.95
    im3 = np.zeros(im.shape)
    for ind in xrange(0,3):
        im3[:, :, ind] = im[:, :, ind] / A[ind]

    transmission = 1 - omega * darkChannel_(im3)
    return transmission
