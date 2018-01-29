import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    直方图:
        1.找点分析
            找直方图，用opencv和numpy函数
            画直方图，用opencv和plt
            原理：
                密度分布信息，x轴总是0-255数值，和相对应y像素值。是针对灰度图。
"""


def histogram_find_plot_analyze():
    """
        1.image:uint8 or float32图片
        2.计算图片的通道索引
        3.mask:None是所有图片。
        4.histSize:BIN数量，[256]
        5.ranges：[0,256]
    :return:
    """
    # opencv
    img = cv2.imread('dog.png')
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # numpy
    hist_np, bins = np.histogram(img.ravel(), 256, [0, 256])
    #    plt.hist(img.ravel(), 256, [0, 256])
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
    plt.show()


def histogram_mask():
    """
        图片某一部分
    :return:
    """
    img = cv2.imread('dog.png', 0)
    # 建立一个mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

    plt.subplot(221), plt.imshow(img, 'gray'), plt.title('original')
    plt.subplot(222), plt.imshow(mask, 'gray'), plt.title('mask')
    plt.subplot(223), plt.imshow(masked_img, 'gray'), plt.title('mask_img')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask), plt.title('img_hidy')
    plt.xlim([0, 256])
    plt.show()


def histogram_equalization():
    """
     图片均衡器概念，提高图片对比度
     高亮度图片会聚集在一起，将像素进行拉伸。常用来提高图片的对比度。
    :return:
    """
    img = cv2.imread('wiki.jpg', 0)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.subplot(122), plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


def after_equalization():
    """

    :return:
    """
    img = cv2.imread('wiki.jpg', 0)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]

    plt.subplot(121), plt.imshow(img2, cmap='gray')
    plt.subplot(122), plt.plot(cdf, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


def equalization_opencv():
    img = cv2.imread('wiki.jpg', 0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))
    cv2.imwrite('res.png', res)
    img_ = cv2.imread('res.png', 0)
    cv2.imshow('hello', img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contrast_limited_adaptive_histogram_equalization():
    """
       局部均衡化
    :return:
    """
    img = cv2.imread('wiki.jpg', 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cll = clahe.apply(img)
    cv2.imwrite('test.png', cll)


def histograms_2D():
    """
        寻找和画2维直方图
        1维中只有一个特征，2维两个特征，通常是色度和饱和度。Hue Saturation
        1.BGR->HSV,channels=[0,1].H and S shape
        2.bins=[180,256] 180 for H ,256 for S 平面
        3. range=[0,180,0,256]
    :return:
    """
    img = cv2.imread('test.jpg')
    # 1.找到直方图
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # numpy
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    img_np = cv2.imread('test.jpg')
    hsv_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    hist_np, xbins_np, ybinx_np = np.histogram2d(h.ravel(), s.ravel(), [180, 256], [[0, 180], [0, 256]])
    # 2.画直方图
    # cv2.imshow()得到180x256数组得到灰度图
    # 用plt显示
    plt.imshow(hist, interpolation='nearest')
    plt.show()

def histogram_back_projection():
    """
        背后投影
            用于图片一部分或者找图片感兴趣的部分
        
    :return:
    """


# histogram_find_plot_analyze()
# histogram_mask()
# histogram_equalization()
# after_equalization()
# equalization_opencv()
histograms_2D()
