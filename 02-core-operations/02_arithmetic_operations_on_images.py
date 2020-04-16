import cv2
import numpy as np
import matplotlib.pylab as plt

"""
目标
    学习图片算术操作：加，减，按操作
    cv2.aa()和cv2.addWeighted()等等
"""


def image_addition():
    """
    1.cv.add()或者res=img1+img2 Numpy方法
        两者之间不同:
            1.opencv相加是饱和操作(saturated operation)
            2.numpy相加是模操作(modulo 操作)
    2.两个图片应该一样的深度和类型，或者第二个图片只是个标量值(scalar value)
    :return:
    """
    x = np.uint8([250])
    y = np.uint8([10])
    print(x, y)
    print(cv2.add(x, y))  # 250+10=260->255
    print(x + y)  # 250+10=260%256=4


# image_addition()


def image_blending():
    """
    也是图片操作，但是图片不同的比重(weight)为了感觉混合或者透明。
        g(x)=(1-α)f0(x) + αf1(x) ，其中α取值0-1.

    cv2.addWeighted()要下面公式:
        dst=α*img1+β*img2+γ
    :return:
    """
    img1 = cv2.imread("image/test1.jpg")
    print(img1[:2, :3, :3])
    img2 = cv2.imread("image/test2.jpg")
    print(img2[:2, :3, :3])
    cv2.imshow('test1', img1)
    cv2.imshow('test2', img2)
    # 小数，四舍五入  末尾
    dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 2)
    print(dst[:2, :3, :3])
    cv2.imshow('dts', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# image_blending()


def bitwise_operations():
    """
     按位操作：AND,OR,NOT 和XOR
        获取图片任何部分，细化和处理  图片矩形区域ROI(rang of image)
    :return:
    """
    # load two images
    img1 = cv2.imread('image/test.jpg')
    img2 = cv2.imread('image/opencv.png')

    # i want to put logo on top-left corner,so i create a ROI
    rows, cols, channels = img2.shape
    print(img2.shape)
    roi = img1[0:rows, 0:cols]

    # now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 阀值: 10为值阀，<10 转化为0  , >10 255
    # 255 为maxVal 指定大于阀的值
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)

    # bitwise_not 255减去原来像素：如原来20,去反为235
    mask_inv = cv2.bitwise_not(mask)

    # mask 相当于去除改部分，mask_inv 保留mask以外的部分
    # now black-out the area of logo in ROI(相当于去掉OpenCV 透明的地方,去掉背景色)
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # take only region of logo from logo image(扣掉OpenCv)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst
    cv2.imshow('res', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# bitwise_operations()


