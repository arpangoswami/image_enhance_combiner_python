import sys
import cv2 as cv
import numpy as np
import uuid
import pywt

# This function does the coefficient fusing according to the fusion method
def fuseCoeff(cooef1, cooef2, method):
    if method == 'mean':
        cooef = (cooef1 + cooef2) / 2
    elif method == 'min':
        cooef = np.minimum(cooef1, cooef2)
    elif method == 'max':
        cooef = np.maximum(cooef1, cooef2)
    else:
        cooef = []
    return cooef


# Params
FUSION_METHOD = 'mean'  # Can be 'min' || 'max || anything you choose according theory


def wavelet_fusion(blur, grayscale):
    print(blur.shape)
    print(grayscale.shape)
    grayscale = cv.resize(grayscale, (blur.shape[1], blur.shape[0]))
    print(blur.shape)
    print(grayscale.shape)
    # Fusion algo
    # First: Do wavelet transform on each image
    wavelet = 'db1'
    cooef1 = pywt.wavedec2(blur[:, :], wavelet)
    cooef2 = pywt.wavedec2(grayscale[:, :], wavelet)
    # Second: for each level in both image do the fusion according to the desire option
    fused_coeff = []
    for i in range(len(cooef1) - 1):

        # The first values in each decomposition is the apprximation values of the top level
        if i == 0:

            fused_coeff.append(fuseCoeff(cooef1[0], cooef2[0], FUSION_METHOD))

        else:

            # For the rest of the levels we have tupels with 3 coeeficents
            c1 = fuseCoeff(cooef1[i][0], cooef2[i][0], FUSION_METHOD)
            c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD)
            c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD)

            fused_coeff.append((c1, c2, c3))

    # Third: After we fused the co-efficient we need to transfer back to get the image
    fused_image = pywt.waverec2(fused_coeff, wavelet)

    # Forth: normalize values to be in uint8
    fused_image = np.multiply(np.divide(
        fused_image - np.min(fused_image), (np.max(fused_image) - np.min(fused_image))), 255)
    fused_image = fused_image.astype(np.uint8)
    return fused_image


# Reading and showing image
path = sys.path[0]+"/resources/Photos/park.jpg"
print("PATH: ",path)
imgPark = cv.imread(path)
cv.imshow("Park-window", imgPark)

# blur
blurredImage = cv.GaussianBlur(imgPark, (7, 7), cv.BORDER_DEFAULT)
cv.imshow("Blurred image", blurredImage)

blurred_image_name = str(uuid.uuid4())
blurred_image_name += ".jpg"
cv.imwrite(blurred_image_name, blurredImage)

# converting to grayscale
grayScalePark = cv.cvtColor(imgPark, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale-park", grayScalePark)
grayscale_image_name = str(uuid.uuid4())
grayscale_image_name += ".jpg"
cv.imwrite(grayscale_image_name, grayScalePark)

(blueBlur, greenBlur, redBlur) = cv.split(blurredImage)

blueFused = wavelet_fusion(blueBlur, grayScalePark)
greenFused = wavelet_fusion(greenBlur, grayScalePark)
redFused = wavelet_fusion(redBlur, grayScalePark)

fusedImage = cv.merge([blueFused, greenFused, redFused])
resized = cv.resize(fusedImage, (imgPark.shape[1], imgPark.shape[0]))
cv.imshow('Resized Fused', resized)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
image_sharp = cv.filter2D(src=fusedImage, ddepth=-1, kernel=kernel)
resized_sharpened = cv.resize(
    image_sharp, (imgPark.shape[1], imgPark.shape[0]))
cv.imshow('Sharpened image', resized_sharpened)
new_image_name = str(uuid.uuid4())
new_image_name += ".jpg"
cv.imwrite(new_image_name, resized_sharpened)

cv.waitKey(0)
