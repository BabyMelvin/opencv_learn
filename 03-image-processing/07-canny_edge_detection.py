import cv2
import numpy as np
import matplotlib.pylab as plt

"""
canny 边缘检测
    canny 边缘检测的概念
    canny边缘检测是一种流行算法。需要多个步骤的算法
    1.减噪(noise reduction)
        由于边缘检测易受噪声影响，第一步减噪。
    2.找到图像强度梯度intensity gradient of the image
      平滑图片，然后用Sobel内核竖直和水平方向获得一阶导数Gx和Gy
      通过两个图片，可以获得边缘梯度和每个像素的方向。
      Edge_Gradient(G)=sqrt(Gx^2+Gy^2)
      Angle（@）=tan^-1*(Gy/Gx)
      梯度方向总是垂直(perpendicular)于边缘。近似于四个角总的一个:水平、竖直和两个对角线方向.
    3.非极大值压制
      获得梯度大小和方向后，全范围扫描为了去除不想要的像素点（可能不是组建边缘的部分）
      为了这样要检查是否本地最大值在梯度相邻的地址。如果最大，则是下一个边缘，不是则为0.
      简而言之，你获得的结果，带有薄边缘的二值图
      
    4.滞后阀值发(hysteresis thresholding)
      这个阶段决定是否是真的边缘。为了实现这个需要两个阀值，最大和最小。
        边缘强度梯度比最大值大，确定是边缘
        边缘强度梯度比最小值小，去定不是边缘，丢弃。
        两线之间，却决与相关性
            如果他们和确定边界连接的像素，认为他们是边缘的一部分。
            否则被丢弃
"""


def canny_edge_detection():
    """
        cv2.Canny()：
         第一个参数：输入图片
         第二个参数和第三个参数：最小值和最大值
         第三个参数：aperture_size，Sobel内核尺寸使用找到图片梯度。
            默认是3
         最后一个参数：L2gradient具体化公式，为了找到梯度大小。
            True使用上面的公式，更精确。
            False:Edge_Gradient(G)=|Gx|+|Gy|(默认)
    :return:
    """
    img = cv2.imread('image/meisi.jpg', 0)
    edges = cv2.Canny(img, 100, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('original'), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edget-canny'), plt.xticks([]), plt.yticks([])
    plt.show()


canny_edge_detection()
