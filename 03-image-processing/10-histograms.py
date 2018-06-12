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
        4.histSize:BIN数量，[256] . Bins取得采样点数目,平均分配[0,256]，然后依次连接各个点
        5.ranges：[0,256]
    :return:
    """
    # opencv
    img = cv2.imread('image/dog.png')
    # hist = cv2.calcHist([img], [0], None, [1000], [0, 256])
    # numpy
    # hist_np, bins = np.histogram(img.ravel(), 256, [0, 256])
    # plt.hist(img.ravel(), 256, [0, 256])
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [1000], [0, 256])
        plt.plot(histr, color=col)
    plt.show()


# histogram_find_plot_analyze()


def histogram_mask():
    """
        图片某一部分
    :return:
    """
    img = cv2.imread('image/dog.png', 0)
    # 建立一个mask ,img.shape[:2] 第一维中的0,1索引。
    mask = np.zeros(img.shape[:2], np.uint8)
    # 255 是白色
    mask[100:300, 100:400] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)  # 取得图像的该区域

    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

    plt.subplot(221), plt.imshow(img, 'gray'), plt.title('original')
    plt.subplot(222), plt.imshow(mask, 'gray'), plt.title('mask')
    plt.subplot(223), plt.imshow(masked_img, 'gray'), plt.title('mask_img')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask), plt.title('img_hidy')
    plt.xlim([0, 256])
    plt.show()


# histogram_mask()


def histogram_equalization():
    """
     图片均衡器概念，提高图片对比度
     高亮度图片会聚集在一起，将像素进行拉伸。常用来提高图片的对比度。
    :return:
    """
    img = cv2.imread('image/wiki.jpg', 0)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    # 　累积和. [1,2,3] 累计和 [1,3,6]
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.subplot(122), plt.plot(cdf_normalized, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


# histogram_equalization()


def after_equalization():
    """

    :return:
    """
    img = cv2.imread('image/wiki.jpg', 0)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    cdf = hist.cumsum()

    # 去掉0值
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]

    plt.subplot(121), plt.imshow(img2, cmap='gray')
    plt.subplot(122), plt.plot(cdf_m, color='b')
    plt.hist(img2.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


# after_equalization()


def equalization_in_opencv():
    img = cv2.imread('image/wiki.jpg', 0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))
    # cv2.imwrite('image/res.png', res)
    # img_ = cv2.imread('image/res.png', 0)
    cv2.imshow('hello', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# equalization_in_opencv()


def contrast_limited_adaptive_histogram_equalization():
    """
       局部均衡化
            图片被小块，称为tiles。默认8x8.
            对每小块进行处理，如果小块中有噪声，就会被放大。避免这个发生contrast limiting。
            如果bin的数目超过40，
    :return:
    """
    img = cv2.imread('image/tsukuba_l.jpg', 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cll = clahe.apply(img)
    cv2.imshow('orign', img)
    cv2.imshow('cll', cll)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# contrast_limited_adaptive_histogram_equalization()


def histograms_2D():
    """
        寻找和画2维直方图
        1维中只有一个特征，2维两个特征，通常是色度和饱和度。Hue Saturation
        1.BGR->HSV,channels=[0,1].H and S shape
        2.bins=[180,256] 180 for H ,256 for S 平面
        3. range=[0,180,0,256]
    :return:
    """
    img = cv2.imread('image/test.jpg')
    # 1.找到直方图
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # numpy
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    img_np = cv2.imread('image/test.jpg')
    hsv_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    hist_np, xbins_np, ybinx_np = np.histogram2d(h.ravel(), s.ravel(), [180, 256], [[0, 180], [0, 256]])
    # 2.画直方图
    # cv2.imshow()得到180x256数组得到灰度图
    # 用plt显示
    plt.imshow(hist, interpolation='nearest')
    plt.show()


# histograms_2D()

# 1.histogram_back_projection
def algorithm_in_numpy():
    """
        背后投影
            用于图片一部分或者找图片感兴趣的部分
    :return:
    """
    # algorithm in numpy
    # 1.要计算颜色直方图主要两个,要找的'M'和要搜索'I' (I中搜索M)
    # roi is the object or region of object we need to find
    roi = cv2.imread('rose_red.png')
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # target is the image we search in
    target = cv2.imread('rose.png')
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    # find the histograms using calcHist.Can be done with np.hsitogram2d also
    M = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    I = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # 2.找到比率R=M/I.然后投射R，例如用R作为调色板并创建一个有可能目标所有像素对应的新图片。
    # 例如B（x,y)=R[h(x,y),s(x,y)]h和s是在像素点(x,y)色度和饱和度。
    # 应用这个条件B(x,y)=min[B(x,y),1]
    R = M / I
    h, s, v = cv2.split(hsvt)
    B = R[h.ravel(), s.ravel()]
    B = np.minimum(B, 1)
    B = B.reshape(hsvt.shape[:2])

    # 3.现在用一个圆盘形卷积，B=D*B,D是圆形内核
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(B, -1, disc, B)
    B = np.uint8(B)
    cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)
    # 4.最大密度位置来获得对象的位置。如果想要图片中某个区域，适当阀值获得更好的效果。
    ret, thresh = cv2.threshold(B, 50, 255, 0)


# 2.histogram_back_projection
def backprojection_in_opencv():
    """
    :return:
    """
    roi = cv2.imread('image/rose_red.png')
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    target = cv2.imread('image/rose.png')
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    # calculating object histogram
    roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    # normalize histogram and apply backprojection
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt], [0, 1], roihist, [0, 180, 0, 256])

    # now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)

    # threshold and binary AND
    ret, thresh = cv2.threshold(dst, 50, 255, 0)
    thresh = cv2.merge(thresh, thresh, thresh)
    res = cv2.bitwise_and(target, thresh)

    res = np.vstack((target, thresh, res))
    cv2.imwrite('res.jpg', res)
    cv2.imshow('messi', ret)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


backprojection_in_opencv()
