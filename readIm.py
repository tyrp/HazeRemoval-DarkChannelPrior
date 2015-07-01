from scipy import misc
import matplotlib.pyplot as plt
import cv2

def readIm_(image=None, ext=None, *args, **kwargs):
    im = cv2.imread(image, 1)
    return im