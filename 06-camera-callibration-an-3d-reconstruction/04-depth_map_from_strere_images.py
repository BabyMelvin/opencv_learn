import cv2
import matplotlib.pylab as plt
import numpy as np


def depth_image():
    """
    :return:
    """
    imgL = cv2.imread('image/sukuba_1.png', 0)
    imgR = cv2.imread('image/tsukuba_r.png', 0)

    stero = cv2.createStereroBM(numDisparties=16, blockSize=15)
    disparity = stero.compute(imgL, imgR)
    plt.imshow(disparity, 'gray')
    plt.show()


depth_image()
