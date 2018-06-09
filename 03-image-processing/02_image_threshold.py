import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    简单阀值
    自适应阀值分割
    多阀值
"""


def simple_threshold():
    """
        像素值大于阀值，分配一个值（可能黑色），小于分配另一值(可能白色)
        第一个参数是灰度图，第二个是划分像素阀值，第三个参数最大阀值
        第四个参数，阀值样式:
            cv2.THRESH_BINARY
            cv2.THRESH_BINARY_INV
            cv2.THRESH_TRUNC
            cv2.THRESH_TOZERO
            cv2.THRESH_TOZERO_INV
    :return:
    """
    img = cv2.imread('image/test.jpg')
    ori = img[:3, :3, :]
    # 大于127 为255.小于127为0  二值就是分界意思
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 大于37截取成37.小于37不变 。截取挤压
    ret, thresh3 = cv2.threshold(img, 37, 255, cv2.THRESH_TRUNC)

    # 阀为0或不是0,和截取区别在于阀值范围为0或者是指定值
    ret, thresh4 = cv2.threshold(img, 48, 255, cv2.THRESH_TOZERO)

    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    print("ori", ori)
    print("th3=trunc", thresh3[:3, :3, :])
    print("th4=tozero", thresh4[:3, :3, :])

    titles = ['origin', 'binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


# simple_threshold()


def adaptive_threshold():
    """
        简单阀值，用全局控制阀值。
        算术计算小的区域的阀值，变化会有更好的效果。
        三个输入，一个输出
        阀值计算方法：
            cv2.ADAPTIVE_THRESH_MEAN_C:相邻区域平局值
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C:高斯窗口的权重值
        block_size:决定相邻区域面积大小
        c:一个常数，平均或者权重相减计算
    :return:
    """
    img = cv2.imread('test.jpg', 0)

    img = cv2.medianBlur(img, 5)
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    titles = ['origin', 'global', 'adaptive mean', 'adaptive gaussian']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def Otsu_binarization():
    """
     retVal:
        自动计算双峰图中阀值，为了给直方图。
        输入噪音图片
            1.用127全局阀值。
            2.直接应用otsu's阀值
            3.用5x5高斯核去噪
    :return:
    """
    img = cv2.imread('noisy.jpg', 0)

    # 全局阀值
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # otus's 阀值
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 高斯过滤后的 otsu阀值
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    images = [img, 0, th1, img, 0, th2, blur, 0, th3]
    titles = ['origin', 'histogram', 'global', 'origin', 'histogram', 'otsu',
              'gaussian', 'histogram', 'otsu']
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()
