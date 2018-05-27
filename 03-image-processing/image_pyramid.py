import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    图片金字塔image pyramids
        学习图片金字塔
        学习创建一种水果
    原理:
        有时需要同一个图片不同的分辨率。提供系列不同分辨率图片，称为图片金字塔
        1.高斯金字塔
            在高斯金字塔图中，高水平(低分辨率)由低水平图(高分辨率)移除连续行和列。
        其中高水平(低分辨率)每个像素由高斯权重基础水平的5个像素组成。
        通过这个过程:
            MXN图片变成了M/2xN/2的图像，变成原图的1/4面积，称为八度。
        2.拉普拉斯金字塔
        
"""


def pyramid():
    """

    :return:
    """
    img = cv2.imread('test.jpg')
    lower_reso = cv2.pyrDown(img)
    higher_reso = cv2.pyrUp(img)

    # TODO 其中img和higher_reso不相等，一旦降低分辨率，将会失去信息。
    cv2.imshow('origin', img)
    cv2.imshow('pyramids', lower_reso)
    cv2.imshow('pyramids_high', higher_reso)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_blend_pyramid():
    """
    :return:
    """
    A = cv2.imread('apple.jpg')
    B = cv2.imread('orange.jpg')

    # 1.A产生高斯金字塔图
    G = A.copy()
    gpA = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # B产生高斯金字塔图
    G = B.copy()
    gpB = [G]
    for i in range(6):
        G = cv2.pyrDown(G)
        gpB.append(G)
    # 2.A产生拉普拉斯金字塔
    lpA = [gpA[5]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # B产生拉普拉斯金字塔
    lpB = [gpB[5]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)
    # 3.现在加上左右各个水平的一半图
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols / s], lb[:, cols / 2:]))
        LS.append(ls)

    # 4.现在重新构建
    ls_ = LS[0]
    for i in range(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    # 5.将图片各一半连接起来
    real = np.hstack((A[:, :cols / 2], B[:, cols / 2:]))
    cv2.imwrite('prymids_blend.jpg', ls_)
    cv2.imwrite('direct_blend.jpg', real)


pyramid()
