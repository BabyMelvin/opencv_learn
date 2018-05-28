import cv2
import numpy as np
import matplotlib.pylab as plt

"""
    目标：
        找图片梯度(gradient)，和边缘
    原理:
        三种梯度过滤或高通过滤：Sobel,Scharr和Laplacian  
    1.Sobel和Scharr Derivatives(衍生品)
        Sobel操作是结合高斯平滑加上不同操作。对噪音更好抵抗力(resistant)
        xorder和yorder导数的方向，也可以ksize的内核尺寸。
            ksize=-1 3x3Scharr过滤使用比3x3Sobel过滤效果更好。
    2.Laplacian Derivatives
         Laplacian给如下关下:变化量=变化量的x,y方向的二阶偏导的和。
         ksize=-1使用下面过滤k=[[0,1,0],[1,-4,1],[0,1,0]]
"""


def image_gradients():
    """
    :return:
    """
    img = cv2.imread("image/dave.jpg", 0)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])

    plt.subplot(222), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

    plt.subplot(223), plt.imshow(sobelx, cmap='gray')
    plt.title('SobelX'), plt.xticks([]), plt.yticks([])

    plt.subplot(224), plt.imshow(sobely, cmap='gray')
    plt.title('SobelY'), plt.xticks([]), plt.yticks([])
    plt.show()


def one_important_matter():
    """
    一个重要的问题：
        上个例子，输出类型是cv2.CV_8U或np.uint8.黑-到-白：转换将会保留正斜率(positive slope)’
        将数据转化成np.uint8类型，负斜率将会为零。简单来说，就是丢掉边界。

        为了检测边界，最好使用高输出类型像:cv2.CV_16S,cv2.CV_64F,获得绝对值，然后转化成cv2.CV_8U.
        下面进行验证：水平Sobel过滤不同的结果
    :return:
    """
    img = cv2.imread('image/box.png', 0)

    # output dtype=cv2.CV_8U
    sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)

    # output dtype=cv2.CV_64F.then take its absolute and convert to cv2.Cv_8U
    sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])

    plt.subplot(132), plt.imshow(sobelx8u, cmap='gray')
    plt.title('sobel cv_8u'), plt.xticks([]), plt.yticks([])

    plt.subplot(133), plt.imshow(sobel_8u, cmap='gray')
    plt.title('sobel abs(cv_64f)'), plt.xticks([]), plt.yticks([])
    plt.show()


# image_gradients()
one_important_matter()
