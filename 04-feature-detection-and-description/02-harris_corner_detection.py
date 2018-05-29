import cv2
import matplotlib.pylab as plt
import numpy as np


def harris_corner_detector():
    """
        cv2.cornerHarris():
            img:输入图片
            blockSize:相邻尺寸(the size of neighbourhood)为了检测转角
            ksize:Sobel导数用的变化量(aperture parameter of Sobel derivative used)
            k:自由参数Harris检测等式中
    :return:
    """
    filename = 'image/images.png'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated(扩大) for making the corners,not important
    dst = cv2.dilate(dst, None)

    # threshold for an optimal value ,it may vary depending on the image
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def corer_with_subpixel_accuracy():
    """
    有时需要找到最大精度的转角。cv2.cornerSunPix()将会找到子像素精度转角检测。
    1.首先找到Harris corners
    2.然后传递这些转角的圆心，去细化他们。Harris转角标记为红色，细化转角标记为绿色像素。
    3.我们必须定义停止迭代(iteration)的条件（criteria），通过迭代次数或者某个要求的精度来停止它。
        还需要定义相邻尺寸搜索转角
    :return:
    """
    filename = 'image/arrow.jpg'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # now draw them
    res = np.hstack((centroids, corners))
    res = np.int0(res)
    img[res[:, 1], res[:, 0]] = [0, 0, 255]
    img[res[:, 3], res[:, 2]] = [0, 255, 0]

    cv2.imshow('chees2', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# harris_corner_detector()
corer_with_subpixel_accuracy()
