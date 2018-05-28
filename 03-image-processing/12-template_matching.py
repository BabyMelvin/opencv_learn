import cv2
import numpy as np
import matplotlib.pylab as plt

"""
模式匹配
    目标：使用模式匹配找到图片中物体
    原理:
        模式匹配是一种搜索和寻找一个模式的位置，在大图中。只是简单的模式图片
        在输入图片上滑动(类似2维卷积)，比较模式额模式图片下的内容。
        输入图片(WXH)和模式图片(wxh)，输出图片尺寸为(W-w+1,H-h+1)
        当得到大小，使用cv2.minMaxLoc()找到最大最小值。
            top-left值，并且(w,h)宽高。
        cv2.TM_SQDIFF作为比较的方法，最小值给最小的匹配
"""


def template_matching():
    img = cv2.imread('image/meisi.jpg', 0)
    img2 = img.copy()
    template = cv2.imread('image/ball.jpg', 0)
    w, h = template.shape[:: -1]

    # All the 6 methods for comparion in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR'
        , 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # apply template matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # if the methods is TM_SQDIFF or TM_SQDIFF_NORMAED,take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[0] + h)

        cv2.rectangle(img, top_left, bottom_right, 255, thickness=2)

        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Match Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)
        plt.show()


def template_matching_with_multiple_object():
    """
        假如出现多次在同一个图片上，cv2.minMaxLoc()不会给多个。
        因此使用阀值Matrio游戏中找到硬币
    :return:
    """
    img_rgb = cv2.imread('image/matrio.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('image/little.jpg', 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 20, 255), 2)

    cv2.imshow('reswin', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


template_matching()
template_matching_with_multiple_object()
