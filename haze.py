from functions import *
import argparse

parser = argparse.ArgumentParser(description='Remove image haze using dark channel prior method')
parser.add_argument('-s','--scale', action="store", dest="scale", default=0.5, type=float, help="Scaling factor for images")
parser.add_argument('-f','--folder', action="store", dest="folder", default='beijing1', help="folder name")
parser.add_argument('-n','--file', action="store", dest="file", default='IMG_8763', help="file name")

args = parser.parse_args()

scalingFactor = args.scale
folder = args.folder
fileName = args.file

def deHaze(filename):
    print 'Loading Image', filename
    imageRGB = misc.imresize(readIm(filename), scalingFactor);
    cv2.imwrite('MyResults/' + fileName + '_imageRGB.jpg', imageRGB)
    imageRGB = imageRGB / 255.0

    print 'Getting Dark Channel Prior'
    darkChannel = getDarkChannel(imageRGB);
    cv2.imwrite('MyResults/' + fileName + '_dark.jpg', darkChannel * 255.0)

    print 'Getting Atmospheric Light'
    atmLight = getAtmLight(imageRGB, darkChannel);

    print 'Getting Transmission'
    transmission = getTransmission(imageRGB, atmLight);
    cv2.imwrite('MyResults/' + fileName + '_transmission.jpg', transmission * 255.0)

    print 'Getting Scene Radiance'
    radiance = getRadiance(atmLight, imageRGB, transmission);
    cv2.imwrite('MyResults/' + fileName + '_radiance.jpg', radiance * 255.0)

    print 'Apply Soft Matting'
    mattedTransmission = performSoftMatting(imageRGB, transmission);
    cv2.imwrite('MyResults/' + fileName + '_refinedTransmission.jpg', mattedTransmission * 255.0)

    print 'Getting Scene Radiance'
    betterRadiance = getRadiance(atmLight, imageRGB, mattedTransmission);
    cv2.imwrite('MyResults/' + fileName + '_refinedRadiance.jpg', betterRadiance * 255.0)

deHaze('Dataset/' + folder + '/' + fileName + '.jpg')