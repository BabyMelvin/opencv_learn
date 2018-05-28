import cv2
import matplotlib.pylab as plt
import numpy as np

"""
目标：
    毛图片使用各种低通过滤
    应用自定义过滤处理图片(2D卷积(convolution) )
"""


def _2D_convolution():
    """
    图片过滤：对于一维信号，图片也可以使用各种低通(LPF)或高通滤波器(HPF)处理。
        LPF低通滤波器：
            去除噪音，图片模糊
        HPF高通滤波：
            找图片边缘
    :return:
    """
    img = cv2.imread("noisy.jpg")

    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)

    plt.subplot(121), plt.imshow(img), plt.title('original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst), plt.title('average')
    plt.xticks([]), plt.yticks([])
    plt.show()


def image_blurring():
    """
    图片顺滑
        图片模糊利用低通过滤卷积内核包含了。用来去噪。
        实际上去除高频内容（噪音和边缘）
        有四种模糊技术大会对边缘模糊处理：
        1.平均（averaging）
            这个利用归一化卷积内核完成。获得算有内核下像素平均值，用这个平局值替换中心元素。
            cv2.blur()和cv2.boxFilter()完成
        2.高斯过滤
            用高斯内核进行过滤。具体化内核宽高，并且是正偶数。
                还有X,Y方向的标准差(standard deviation),sigmaX,sigmaY.
        3.median filter
            中值内核.
            很好的去除salt-and-pepper (盐和胡椒)噪音
            高斯内核和归一化内核，替换的值，有可能原图片不存在改值。但是中值就不会出现
        4.bilateral filter
            双边滤波：
                前面是会模糊边缘，顾使用双边滤波。对边缘保留效果很好，但是速度比其他的慢一些。

    :return:
    """
    # 1. averaging
    img = cv2.imread('noisy.jpg')
    blur = cv2.blur(img, (5, 5))
    # 2. Gaussian filter
    blur_gau = cv2.GaussianBlur(img, (5, 5), 0)
    # 3.median filtering
    # 50%的噪音对于原图片
    median = cv2.medianBlur(img, 5)

    # 4.bilateral
    blur_bilteral = cv2.bilateralFilter(img, 9, 75, 75)
    plt.subplot(231), plt.imshow(img), plt.title('original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(blur), plt.title('blurred')
    plt.subplot(233), plt.imshow(blur_gau), plt.title('blur_gau')
    plt.xticks([]), plt.yticks([])
    plt.subplot(234), plt.imshow(median), plt.title('median')
    plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(median), plt.title('median')
    plt.xticks([]), plt.yticks([])
    plt.show()


_2D_convolution()
image_blurring()
