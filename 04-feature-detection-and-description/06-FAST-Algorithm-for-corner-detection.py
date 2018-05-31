import cv2
import matplotlib.pylab as plt
import numpy as np


def fast_feature_detector():
    """
        能够调用任何形式特征点检测。可以具体话阀值，是否极值抑制。
        相邻，有三种标志：
            cv2.FAST_FEATURE_DETECTOR_TYPE_5_6
            cv2.FAST_FEATURE_DETECTOR_7_12
            cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
    :return:
    """
    img = cv2.imread('image/cube.jpg', 0)

    # initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(img, None)

    imgCy = img.copy()

    img2 = cv2.drawKeypoints(img, kp, outImage=imgCy, color=(255, 0, 0))
    img3 = cv2.drawKeypoints(img, kp, outImage=imgCy, color=(0, 255, 0), flags=cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)

    # print all default params
    plt.subplot(221), plt.imshow(imgCy), plt.show()
    plt.subplot(222), plt.imshow(img2), plt.show()
    plt.subplot(223), plt.imshow(img3), plt.show()


fast_feature_detector()
