import cv2
import numpy as np

"""
    几何转化：平移，旋转，affine(放射)
    cv2.wrapAffine():
        2x3转换矩阵
    cv2.warpPerspective()
        3x3转换矩阵
"""


def scaling():
    img = cv2.imread('test.jpg')
    res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

