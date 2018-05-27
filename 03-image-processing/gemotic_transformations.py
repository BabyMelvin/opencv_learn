import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    几何转化：平移，旋转，affine(放射)
    cv2.wrapAffine():
        2x3转换矩阵
    cv2.warpPerspective()
        3x3转换矩阵
"""


def scaling():
    """
        1.scaling
            改变图片尺寸，可以手动设置图片尺寸。
            不同的插补方法(interpolation method)
                shrink：缩小用cv2.INTER_AREA,cv2.INTER_CUBIC（slow）
                zooming:cv2.INTER_LINER(默认）
        2.
    :return:
    """
    img = cv2.imread('test.jpg')
    # res = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # 或者
    height, width = img.shape[:2]
    print(height, width)
    res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
    # plt.subplot对自动对齐大小，看不出效果
    cv2.imshow('image', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def translation():
    """
        改变物体位置。
            如果知道移动(shift）方向（x,y)可以创建这样的转移矩阵
                    M=[[1,0,x],[0,1,y]]

    :return:
    """
    img = cv2.imread('test.jpg', 0)
    rows, cols = img.shape
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotation():
    """
        中心旋转：
            角α，转移矩阵M=[[cosα，-sinα],[sinα，cosα]]
        可调整中心的旋转
            M=[[α，β，(1-α).center.x-β.center.y],[-β,α,β.center.x+(1-α).center.y]]
            α=scale.cosO
            β=scale.sinO
    :return:
    """
    img = cv2.imread('test.jpg', 0)
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('rotation', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def affine_transformation():
    """
       仿射变换：所有平行线输出还是平行线。
       找到旋转矩阵，输入图像的三个点，和对应输出的三个点。
    :return:
    """
    img = cv2.imread('test.jpg')
    rows, cols, ch = img.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))

    plt.subplot(121), plt.imshow(img), plt.title('input')
    plt.subplot(122), plt.imshow(dst), plt.title('output')
    plt.show()


def perspective_transformation():
    """
        透视图，需要3x3矩阵，直线仍然是直线。
        矩阵，需要4个点在输入图片和对应的输出图片。
            4点中3点不能共线
    :return:
    """
    img = cv2.imread('test.jpg')
    rows, cols, ch = img.shape
    pts1 = np.float32([[56, 56], [246, 145], [27, 48], [200, 300]])
    pts2 = np.float32([[0, 0], [200, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (300, 300))

    plt.subplot(121), plt.imshow(img), plt.title('input')
    plt.subplot(122), plt.imshow(dst), plt.title('output')
    plt.show()


# scaling()
# translation()
# rotation()
# affine_transformation()
perspective_transformation()
