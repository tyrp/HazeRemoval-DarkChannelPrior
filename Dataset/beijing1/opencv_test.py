import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('IMG_8763.jpg')
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 2,(255,255,255),2,cv2.CV_AA)
b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])
kernel = np.ones((5,5), np.float32)/25
dst = cv2.filter2D(img, -1, kernel)
# plt.subplot(121); plt.imshow(img); plt.title('bgr')
# plt.subplot(122); plt.imshow(img2); plt.title('rgb')
# plt.show()

cv2.imshow('original', img)
cv2.imshow('altered', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('IMG_8763_GRAY.jpg', img)

def atmLight_(im=None,JDark=None,*args,**kwargs):
    # varargin = cellarray(args)
    # nargin = 2-[im,JDark].count(None)+len(args)

    height, width, __ = im.shape
    imsize = width * height
    numpx = math.floor(imsize / 1000.0)
    JDarkVec = np.reshape(JDark, (imsize, 1))
    ImVec = np.reshape(im, (imsize, 3))
    JDarkVec = np.sort(JDarkVec)
    indices = np.argsort(JDarkVec)
    # print imsize + (-imsize + numpx - 1)
    indices = indices[-(numpx - 1):]
    # print indices
    atmArray = zeros_(1, 3)
    atmSum = 0
    tempArray = zeros_(1, 3)
    tempSum = 0
    windowWidth = 9 #arbitrary
    for ind in xrange(0, int(numpx)-1):
    	for i in xrange(0, windowWidth):
    		tempArray = tempArray + ImVec[indices[ind] + (width * i) - (windowWidth / 2 ) : indices[ind] + (width * i ) + (windowWidth / 2), :]
        tempSum = np.sum(tempArray)
        if tempSum > atmSum:
        	atmArray = tempArray
        	atmSum = tempSum
    A = atmArray/(windowWidth**2)
    return A


def transmissionEst(im=None, A=None,*args,**kwargs):

    omega = 0.95
    # print size_(im)
    # print size_(im)[0]
    # print im.shape
    im3 = np.zeros(im.shape)
    for ind in xrange(0,2):
        im3[:, :, ind] = im[:, :, ind] / A[ind]

    # print im3
    # im3 = im3 / 255.0
    transmission = 1 - omega * darkChannel_(im3)
    # print transmission
    return transmission


def getLaplacian(im=None, *args, **kwargs):

	kd = 0
	windowWidth = 9 #arbitrary
	wk = windowWidth ** 2
	mean = np.zeros(1, 3)
	covMat = np.zeros(3, 3)
	height, width, __ = im.shape
	imsize = width * height
	ImVec = np.reshape(im, (imsize, 3))
	u3 = np.identity(3)
	regParam = 0 #modifiable
	L = np.zeros(imsize, imsize)
	for i in xrange(0, imsize):
		for j in xrange(0, imsize):
			tempWidth = i/imsize
			tempHeight = tempWidth*imsize - i;
			for m in xrange(-windowWidth/2, windowWidth / 2 + 1):
				for n in xrange(-windowWidth/2, windowWidth/2 + 1):
					mean = mean + ImVec[((tempWidth + m) * width) + tempHeight + n, :]
			mean = mean / wk
			X = np.vstack((ImVec[i, :], mean))
			covMat = np.cov(X, rowvar=0)
			kd = 1 if i == j
			temp1 = (ImVec[i, :] - mean).T
			temp2 = np.linalg.inv((covMat + (regParam / wk) * u3))
			temp3 = (ImVec[j, :] - mean)
			temp4 = 1 + temp1 * temp2 * temp3
			result = kd - (1/wk)*temp4
			L[i,j] = result

