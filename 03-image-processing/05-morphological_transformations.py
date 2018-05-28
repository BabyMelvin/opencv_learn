import cv2
import numpy as np
import matplotlib.pylab as plt

"""
形态变换
    将会学习不同的形态操作，腐蚀(erosion)，扩大(dilation)，开放(opening)，关闭(closing)
    原理：
        形态变换随图片shape进行操作。通常对二值(binary images)图片操作.
        需要两个输入，一个原图，第二个`structuring element`或`kernel`决定操作属性。
    两个基本操作是腐蚀和扩大。 然后是一些变化形式opening closing gradient等等
"""


def my_erosion():
    """
    侵蚀基本思想向土壤侵蚀一样，总是侵蚀前景物体的边缘(总是前景保持白色)
    内核函数滑动时候，所有唯一则是1，否则为0.iterations 迭代
    :return:
    """
    img = cv2.imread('j.png')
    kernel = np.ones((5, 5), np.uint8)
    # 1. erosion
    erosion = cv2.erode(img, kernel, iterations=1)
    # 2.dilation
    dialtion = cv2.dilate(img, kernel, iterations=1)
    # 3. opening 去除外部点
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # 4. closing 去除内部点
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # 5.渐变 gradient 类似下划线
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    # 6.top hat下面的内核变为9x9 白色交点位置
    kernel9 = np.ones((9, 9), np.uint8)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel9)
    # 7.black hat    黑色交点位置
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel9)
    plt.subplot(241), plt.imshow(img), plt.title('original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(242), plt.imshow(erosion), plt.title('erosion')
    plt.xticks([]), plt.yticks([])
    plt.subplot(243), plt.imshow(dialtion), plt.title('dialtion')
    plt.xticks([]), plt.yticks([])
    plt.subplot(244), plt.imshow(opening), plt.title('opening')
    plt.xticks([]), plt.yticks([])
    plt.subplot(245), plt.imshow(closing), plt.title('closing')
    plt.xticks([]), plt.yticks([])
    plt.subplot(246), plt.imshow(gradient), plt.title('gradient')
    plt.xticks([]), plt.yticks([])
    plt.subplot(247), plt.imshow(tophat), plt.title('tophat')
    plt.xticks([]), plt.yticks([])
    plt.subplot(248), plt.imshow(blackhat), plt.title('blackhat')
    plt.xticks([]), plt.yticks([])
    plt.show()


def structuring_element():
    """
    我们手动创建结构元素在numpy帮助下,是正方形外形。但是，某些情形，需要圆形或者椭圆
    形的内核。为了这个目的，OpenCv有这个功能函数只需要传递形状和内核尺寸，就得到想要的内核。

    :return:
    """
    print(cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    print(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    print(cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))


# my_erosion()
structuring_element()
