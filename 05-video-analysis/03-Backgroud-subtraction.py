import cv2
import matplotlib.pylab as plt
import numpy as np


def backgroud_subtractor_mog():
    """
    当编程需要创建一个背景对象函数cv2.createBackgroundSubtractorMOG()
    有一些可选的参数：
        历史长度、高斯混合数目、阀值等
    backgroundsubtrator.apply()获得前景mask
    :return:
    """
    cap = cv2.VideoCapture('image/vtest.avi')
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while (1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)
        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def background_subtractor_mog2():
    """
    有个选择，阴影是否被检测到：detectShadows=True（默认），检测标记阴影，减少速度。
    阴影将会标记为灰色
    :return:
    """
    cap = cv2.VideoCapture('image/vtest.avi')
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    while (1):
        ret, frame = cap.read()
        fgmask = fgbg.apply(frame)
        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff

        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def background_subtractor_GMG():
    cap = cv2.VideoCapture('vtest.avi')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.createBackgroundSubtractorGMG()
    while (1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        cv2.imshow('frame', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


# backgroud_subtractor_mog()
# background_subtractor_mog2()
background_subtractor_GMG()
