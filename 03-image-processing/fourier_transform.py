import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    目标:
        1. 寻找图片中傅里叶变换
        2. 利用numpy中的FFT
        3. 傅里叶变换的一些应用
    原理：
        傅里叶变换是用来分析频域特征
"""


def fourier_transform():
    """
    :return:
    """
    img = cv2.imread('test.jpg', 0)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('magnitude spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


fourier_transform()
